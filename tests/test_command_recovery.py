"""Offline regression tests: inject transport errors, never call a real provider."""
import asyncio
import json
import logging
import ssl
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from e2b import CommandExitException, FileNotFoundException
from agent_snapshot_resume.sandbox.snapshot_env import SnapshotCloneEnvironment


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def env(self, first_error=None, code=0, done_on_fault=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        state = {'done': False}
        async def wait_first():
            if first_error:
                state['done'] = done_on_fault
                raise first_error
            state['done'] = True
            return SimpleNamespace(stdout='stream output', stderr='', exit_code=code)
        async def wait_second():
            state['done'] = True
            if code:
                raise CommandExitException(stdout='', stderr='', exit_code=code, error='failed')
            return SimpleNamespace(stdout='', stderr='', exit_code=0)
        h = SimpleNamespace(pid=1234, wait=AsyncMock(side_effect=wait_first), disconnect=AsyncMock())
        h2 = SimpleNamespace(pid=1234, wait=AsyncMock(side_effect=wait_second), disconnect=AsyncMock())
        async def read(path, **kwargs):
            if path.endswith('/exit'):
                if not state['done']:
                    raise FileNotFoundException('not ready')
                return str(code)
            if path.endswith('/stdout'): return 'all stdout before and after disconnect\n'
            if path.endswith('/stderr'): return 'all stderr\n'
            raise AssertionError(path)
        sb = SimpleNamespace(sandbox_id='test-sandbox',
                             commands=SimpleNamespace(run=AsyncMock(return_value=h), connect=AsyncMock(return_value=h2)),
                             files=SimpleNamespace(read=AsyncMock(side_effect=read)))
        e = object.__new__(SnapshotCloneEnvironment)
        e._sandbox = sb
        e._resolve_user = lambda u: u or 'root'
        e._merge_env = lambda v: v
        e._dispatch_command = AsyncMock(return_value=h)  # old implementation reaches this in RED
        e.task_env_config = SimpleNamespace(workdir='/app')
        e._workdir = '/app'
        e.session_id = 'test__env'
        e.trial_paths = SimpleNamespace(trial_dir=Path(temp.name))
        e.logger = logging.getLogger('recovery-test')
        e._recovery_delay = 0.001
        return e, sb, state

    async def test_ssl_reconnects_same_pid_once_and_restores_full_output(self):
        e,s,_ = self.env(ssl.SSLError('record layer failure'))
        r = await e.exec('some non-idempotent command', timeout_sec=10)
        self.assertEqual(r.stdout, 'all stdout before and after disconnect\n')
        self.assertEqual(r.stderr, 'all stderr\n')
        self.assertEqual(r.return_code, 0)
        self.assertEqual(s.commands.run.await_count, 1)
        self.assertEqual(s.commands.connect.call_args.args, (1234,))
        self.assertLessEqual(s.commands.connect.call_args.kwargs['timeout'],10)

    async def test_nonzero_exit_is_not_retried(self):
        e,s,_ = self.env(ssl.SSLError('record layer failure'), code=7)
        r = await e.exec('exit 7', timeout_sec=10)
        self.assertEqual(r.return_code, 7)
        self.assertEqual(s.commands.run.await_count,1)

    async def test_finished_during_disconnect_uses_receipt(self):
        e,s,_ = self.env(ssl.SSLError('record layer failure'),done_on_fault=True)
        r = await e.exec('short task',timeout_sec=10)
        self.assertEqual(r.return_code,0)
        self.assertEqual(s.commands.connect.await_count,0)

    async def test_dispatch_ambiguity_never_retries(self):
        e,s,_ = self.env()
        s.commands.run.side_effect=ssl.SSLError('ambiguous start')
        e._dispatch_command.side_effect=ssl.SSLError('ambiguous start')
        with self.assertRaises(ssl.SSLError): await e.exec('side effect',timeout_sec=10)
        self.assertEqual(s.commands.connect.await_count,0)
        self.assertLessEqual(s.commands.run.await_count,1)

    async def test_unrelated_valueerror_propagates(self):
        e,s,_=self.env(ValueError('bad input'))
        with self.assertRaisesRegex(ValueError,'bad input'):await e.exec('task',timeout_sec=10)
        self.assertEqual(s.commands.connect.await_count,0)

    async def test_deadline_not_extended_by_reconnect(self):
        e,s,_=self.env(ssl.SSLError('record layer failure'))
        s.commands.connect.side_effect=ssl.SSLError('still broken')
        with self.assertRaises(TimeoutError):await e.exec('task',timeout_sec=0.03)
        self.assertEqual(s.commands.run.await_count,1)

    async def test_cancellation_does_not_restart(self):
        e,s,_=self.env(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):await e.exec('task',timeout_sec=10)
        self.assertEqual(s.commands.connect.await_count,0)


if __name__=='__main__':unittest.main()
