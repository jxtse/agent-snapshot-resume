"""Scoped E2B command recovery: one dispatch, same-PID reconnect, durable outputs.

A long agent command (e.g. the resumed `claude --print` run) must survive a broken RPC stream without being
re-executed: re-running a non-idempotent command would change the environment the agent continues in.
`SnapshotCommandRecovery.exec` therefore
  * dispatches the command exactly once (SDK background mode); an unacknowledged dispatch is NOT retried;
  * redirects stdout/stderr and the exit code into a private remote directory (durable receipt);
  * on a transport error, reconnects to the SAME pid (never redispatches) within a bounded window;
  * keeps the caller's deadline (reconnects never extend it);
  * journals every step to <trial_dir>/command-recovery/<session_id>/<token>.json.
No SDK monkeypatch and no change to other Harbor environments.
"""
import asyncio
import hashlib
import json
import shlex
import time
import uuid

from e2b import CommandExitException, FileNotFoundException
from harbor.environments.base import ExecResult
from harbor.environments.e2b import effective_exec_cwd


def transport_error(exc):
    return (type(exc).__name__ in {'SSLError', 'ReadError', 'ReadTimeout', 'RemoteProtocolError',
                                  'ConnectError', 'ConnectTimeout', 'SandboxException', 'NotFoundException'}
            or (isinstance(exc, ValueError) and 'EnvelopeFlags' in str(exc))
            or str(exc) == 'Command ended without an end event')


class SnapshotCommandRecovery:
    _recovery_delay = 2.0
    _recovery_window = 300.0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self._sandbox is None:
            raise RuntimeError('Environment not started')
        started = time.monotonic()
        deadline = started + timeout_sec if timeout_sec else None
        token = uuid.uuid4().hex
        remote = '/tmp/.asr-exec-' + token
        journal_dir = self.trial_paths.trial_dir / 'command-recovery' / self.session_id
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal = journal_dir / (token + '.json')
        state = {'sandbox_id': self._sandbox.sandbox_id, 'remote': remote, 'pid': None,
                 'command_sha256': hashlib.sha256(command.encode()).hexdigest(),
                 'timeout_sec': timeout_sec, 'started_at': time.time(), 'status': 'dispatch_intent', 'events': []}

        def save():
            journal.write_text(json.dumps(state, indent=2) + '\n')

        def event(kind, **values):
            state['events'].append({'event': kind, 'elapsed_s': time.monotonic()-started, **values})
            save()

        q = shlex.quote
        # bash -c is the provider's native execution shell. Keep original cwd,
        # environment and command bytes; only detach stdout from the RPC stream.
        wrapper = (f'mkdir -m 700 {q(remote)} && {{ '
                   f'bash -c {q(command)} > {q(remote + "/stdout")} 2> {q(remote + "/stderr")}; '
                   f'rc=$?; printf "%s" "$rc" > {q(remote + "/exit.tmp")}; '
                   f'mv {q(remote + "/exit.tmp")} {q(remote + "/exit")}; exit "$rc"; }}')
        user = self._resolve_user(user)
        user = str(user) if user is not None else 'root'
        handle = None
        save()

        async def read(name):
            return await self._sandbox.files.read(remote + '/' + name, user=user, request_timeout=30)

        async def receipt():
            try:
                return int((await read('exit')).strip())
            except FileNotFoundException:
                return None

        async def run():
            nonlocal handle
            # Do NOT use Harbor's decorated dispatch retry: a missing acknowledgement
            # is ambiguous. Persist intent and leave that failure for reconciliation.
            handle = await self._sandbox.commands.run(
                cmd=wrapper, background=True,
                cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
                envs=self._merge_env(env), timeout=timeout_sec or 0, user=user)
            pid = handle.pid
            state.update(pid=pid, status='running')
            save()
            recovering_since = None
            completed_code = None
            while True:
                if deadline and time.monotonic() >= deadline:
                    raise TimeoutError('Original command deadline exhausted')
                try:
                    if handle is None:
                        code = await receipt()
                        if code is not None:
                            event('completion_receipt_after_disconnect', exit_code=code)
                            break
                        remaining = max(0.001, deadline-time.monotonic()) if deadline else 0
                        handle = await self._sandbox.commands.connect(pid, timeout=remaining, request_timeout=30)
                        if handle.pid != pid:
                            raise RuntimeError('Reconnected command PID mismatch')
                        event('reconnected_same_pid', pid=pid)
                    try:
                        result = await handle.wait()
                        completed_code = result.exit_code
                    except CommandExitException as exc:
                        completed_code = exc.exit_code
                    code = await receipt()
                    if code is None:
                        raise RuntimeError('Command ended without durable exit receipt')
                    if completed_code != code:
                        raise RuntimeError('Command exit status and durable receipt disagree')
                    break
                except Exception as exc:
                    if not transport_error(exc):
                        raise
                    event('transport_interruption', exception_type=type(exc).__name__, pid=pid)
                    if recovering_since is None:
                        recovering_since = time.monotonic()
                    if time.monotonic() - recovering_since >= self._recovery_window:
                        raise RuntimeError('Same-PID recovery window exhausted; never redispatch') from exc
                    if handle is not None:
                        try:
                            await handle.disconnect()
                        except Exception:
                            pass
                    handle = None
                    delay = self._recovery_delay
                    if deadline:
                        delay = min(delay, max(0, deadline-time.monotonic()))
                    await asyncio.sleep(delay)
            # Idempotent file reads can retry without re-running a command.
            for n in range(5):
                try:
                    stdout, stderr = await read('stdout'), await read('stderr')
                    state.update(status='completed', return_code=code, elapsed_s=time.monotonic()-started)
                    save()
                    return ExecResult(stdout=stdout, stderr=stderr, return_code=code)
                except Exception as exc:
                    if n == 4 or not transport_error(exc):
                        raise
                    event('output_read_retry', exception_type=type(exc).__name__)
                    await asyncio.sleep(self._recovery_delay)

        try:
            if deadline:
                async with asyncio.timeout_at(deadline):
                    return await run()
            return await run()  # Harbor still enforces the original enclosing agent deadline.
        except BaseException as exc:
            state.update(status='interrupted', exception_type=type(exc).__name__, elapsed_s=time.monotonic()-started)
            save()
            raise
        finally:
            if handle is not None:
                try:
                    await handle.disconnect()
                except Exception:
                    pass
