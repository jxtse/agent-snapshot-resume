"""Live sandbox smoke test of command recovery (no model, no key). Skipped unless ASR_LIVE_TESTS=1.

Needs E2B_API_KEY / E2B_API_URL / E2B_DOMAIN and ASR_LIVE_TEMPLATE (a template or snapshot id to start from).
Creates ONE sandbox tagged owner=<ASR_OWNER_TAG>-recovery-test and kills it at the end.
"""
import asyncio
import logging
import os
import ssl
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_snapshot_resume.sandbox.command_recovery import SnapshotCommandRecovery

pytestmark = pytest.mark.live


class FaultHandle:
    def __init__(self, real, delay):
        self.real, self.pid, self.delay = real, real.pid, delay

    async def wait(self):
        await asyncio.sleep(self.delay)
        await self.real.disconnect()
        raise ssl.SSLError('injected record layer failure after acknowledged process start')

    async def disconnect(self):
        await self.real.disconnect()


class InjectedCommands:
    def __init__(self, real, delay):
        self.real, self.delay, self.starts, self.connects = real, delay, [], []

    async def run(self, **kw):
        h = await self.real.run(**kw)
        self.starts.append(h.pid)
        return FaultHandle(h, self.delay) if self.delay is not None else h

    async def connect(self, pid, **kw):
        self.connects.append(pid)
        return await self.real.connect(pid, **kw)


class Env(SnapshotCommandRecovery):
    def _resolve_user(self, u):
        return u or 'root'

    def _merge_env(self, v):
        return v


def test_live_recovery():
    from e2b import AsyncSandbox
    from agent_snapshot_resume.config import load_env_file, owner_tag

    load_env_file()
    template = os.environ.get("ASR_LIVE_TEMPLATE")
    if not template:
        pytest.skip("set ASR_LIVE_TEMPLATE")
    owner = owner_tag() + "-recovery-test"

    async def main():
        sb = await AsyncSandbox.create(template=template, timeout=600, metadata={'owner': owner},
                                       allow_internet_access=True)
        try:
            with tempfile.TemporaryDirectory() as td:
                for label, delay, sleep, code in [('live-disconnect', 1, 6, 7), ('already-finished', 3, 0, 0),
                                                  ('normal', None, 0, 0)]:
                    cmd = InjectedCommands(sb.commands, delay)
                    e = Env()
                    e._sandbox = SimpleNamespace(sandbox_id=sb.sandbox_id, commands=cmd, files=sb.files)
                    e.trial_paths = SimpleNamespace(trial_dir=Path(td))
                    e.session_id = label
                    e.task_env_config = SimpleNamespace(workdir='/tmp')
                    e._workdir = '/tmp'
                    e.logger = logging.getLogger('live-recovery')
                    e._recovery_delay = .1
                    marker = '/tmp/recovery-once-' + label
                    command = (f'printf "once\\n" >> {marker}; printf "before:%s:%s\\n" "$PWD" "$RECOVERY_TEST"; '
                               f'printf "stderr-before\\n" >&2; sleep {sleep}; '
                               f'printf "after\\n"; printf "stderr-after\\n" >&2; exit {code}')
                    r = await e.exec(command, env={'RECOVERY_TEST': 'preserved'}, timeout_sec=60, user='root')
                    assert (await sb.files.read(marker)).splitlines() == ['once']
                    assert len(cmd.starts) == 1
                    assert r.stdout == 'before:/tmp:preserved\nafter\n', repr(r.stdout)
                    assert r.stderr == 'stderr-before\nstderr-after\n', repr(r.stderr)
                    assert r.return_code == code
                    if label == 'live-disconnect':
                        assert len(cmd.connects) >= 1
                    if label == 'already-finished':
                        assert len(cmd.connects) == 0
        finally:
            await sb.kill()

    asyncio.run(main())
