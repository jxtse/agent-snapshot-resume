"""SnapshotCloneEnvironment: Harbor environment whose AGENT sandbox is cloned from a snapshot.

Job config:
    environment:
      import_path: agent_snapshot_resume.sandbox.snapshot_env:SnapshotCloneEnvironment
      kwargs: {snapshot_id, expected_fingerprint, fingerprint_command, expected_auxiliary_hashes,
               expected_verifier_hashes}

* The agent sandbox (Harbor session id ending in "__env") is created FROM the snapshot (environment state at step k,
  made by `asr prepare-snapshot`) instead of the task template. Clones cannot take create-time env vars on the FC
  service (HTTP 400), so none are passed; per-command env still works.
* Every other sandbox (the separate verifier, "<trial>__verifier__...") is created normally from its own template.
* Fail-closed gates before the agent starts:
    expected_fingerprint       output of `fingerprint_command` in the clone == the value recorded on the source
    expected_auxiliary_hashes  sha256 of restored auxiliary files (e.g. background-task outputs)
    expected_verifier_hashes   sha256 of the verifier's /tests files == the local task's tests/ (checked in the
                               verifier sandbox)
  Hash checks are written to <trial_dir>/infra-validation/.
* exec() goes through SnapshotCommandRecovery (one dispatch, same-PID reconnect).
* A verifier image that ships an incomplete /tests (only the top-level test files) is repaired from the task's own
  tests/ before grading, never by hand: the gate's expectation *is* the task's tests/, so restoring it is the fix.
"""
import json
from pathlib import Path

from e2b import AsyncSandbox
from harbor.models.task.config import NetworkMode

from ..config import owner_tag
from ..replay.fingerprint import DEFAULT_ROOTS, build_fingerprint_command
from ..replay.gate import check_hashes
from .command_recovery import SnapshotCommandRecovery
from .fc_env import SANDBOX_TIMEOUT_SEC, SandboxEnvironment
from .runtime_hash import collect_command, hash_problems, parse_collect, tests_files_tar_bytes

UPLOAD_PART_BYTES = 8 * 1024 * 1024

DEFAULT_FINGERPRINT = build_fingerprint_command(list(DEFAULT_ROOTS))


class SnapshotCloneEnvironment(SnapshotCommandRecovery, SandboxEnvironment):
    def __init__(self, *args, snapshot_id: str | None = None, expected_fingerprint=None,
                 expected_auxiliary_hashes=None, expected_verifier_hashes=None, fingerprint_command=None,
                 owner: str | None = None, job: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_id = snapshot_id
        self._expected_fingerprint = expected_fingerprint
        self._fingerprint_command = fingerprint_command or DEFAULT_FINGERPRINT
        self._expected_auxiliary_hashes = expected_auxiliary_hashes or {}
        self._expected_verifier_hashes = expected_verifier_hashes or {}
        self._owner = owner or owner_tag()
        self._job = job
        # Harbor names the agent env "<trial>__env"; the separate verifier env gets "<trial>__verifier__<key>"
        # (possibly hash-truncated), so anything not ending in "__env" is built from its own template.
        self._is_verifier = not (self.session_id or "").endswith("__env")

    async def start(self, force_build: bool):
        if self._is_verifier or not self._snapshot_id:
            await super().start(force_build)
            if self._expected_verifier_hashes:
                await self._check_runtime_hashes(self._expected_verifier_hashes, 'verifier')
            return
        await self._create_sandbox()
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self.exec("chmod 1777 /tmp", user="root")
        if self._expected_fingerprint:
            r = await self.exec(getattr(self, '_fingerprint_command', DEFAULT_FINGERPRINT), timeout_sec=600, user='root')
            if r.return_code != 0 or r.stdout.strip() != self._expected_fingerprint.strip():
                raise RuntimeError('Restored task fingerprint does not match the gated source')
        if self._expected_auxiliary_hashes:
            await self._check_runtime_hashes(self._expected_auxiliary_hashes, 'auxiliary')

    async def _check_runtime_hashes(self, expected, label):
        """Fail-closed check of restored files, with the guest's per-file outcome recorded and reported.

        A verifier environment is repaired once from the task's own tests/ when the check finds files the image does
        not carry (some task images ship /tests without its nested data dirs), then the check runs again: the trial
        must be graded on the task's real tests, not fail because of how an image was packaged.
        """
        expected = dict(expected)
        actual, errors = await self._collect_runtime_hashes(expected)
        repaired = None
        if label == 'verifier' and hash_problems(expected, actual, errors):
            repaired = await self._repair_verifier_tests(expected)
            if repaired.get('files'):
                actual, errors = await self._collect_runtime_hashes(expected)
        record = {'sandbox_id': self._sandbox.sandbox_id, 'label': label, 'expected': expected,
                  'actual': actual, 'errors': errors, 'repaired': repaired}
        target = self.trial_paths.trial_dir / 'infra-validation'
        target.mkdir(parents=True, exist_ok=True)
        (target / (self.session_id + '-' + label + '.json')).write_text(json.dumps(record, indent=2))
        problems = hash_problems(expected, actual, errors)
        if problems:
            raise RuntimeError(f'{label} runtime files do not match the task definition: ' + '; '.join(problems))
        return check_hashes(expected, actual, label)

    async def _collect_runtime_hashes(self, expected):
        result = await self.exec(collect_command(expected), timeout_sec=180, user='root')
        if result.return_code != 0:
            return {}, {p: 'collector failed: ' + (result.stderr or '')[-300:] for p in expected}
        return parse_collect(result.stdout or '{}')

    async def _repair_verifier_tests(self, expected):
        """Restore the expected /tests files in this verifier sandbox from the task's tests/ (the gate's own
        expectation): only the expected paths, so a task whose image deliberately omits parts of tests/ (packed
        archives a build step unpacks) keeps its corpus exactly as the image defines it."""
        src = Path(self.environment_dir)
        rels = sorted({p.split('/tests/', 1)[1] for p in expected if p.startswith('/tests/')})
        if not src.is_dir() or not rels:
            return {'files': 0, 'reason': f'no local tests/ at {src}'}
        data, n = tests_files_tar_bytes(src, rels)
        remote = '/tmp/.asr-tests-repair.tar.gz'
        await self._upload_bytes(remote, data)
        r = await self.exec(f'tar -xzf {remote} -C /tests && rm -f {remote}', timeout_sec=600, user='root')
        if r.return_code != 0:
            return {'files': 0, 'source': str(src), 'reason': (r.stderr or '')[-500:]}
        return {'files': n, 'bytes': len(data), 'source': str(src)}

    async def _upload_bytes(self, remote, data):
        """Write a guest file, in parts: one API write cannot carry a 25 MB tests archive."""
        if len(data) <= UPLOAD_PART_BYTES:
            await self._sandbox.files.write(remote, data)
            return
        parts = []
        for i, lo in enumerate(range(0, len(data), UPLOAD_PART_BYTES)):
            part = f'{remote}.part{i}'
            await self._sandbox.files.write(part, data[lo:lo + UPLOAD_PART_BYTES])
            parts.append(part)
        r = await self.exec('cat ' + ' '.join(parts) + ' > ' + remote + ' && rm -f ' + ' '.join(parts),
                            timeout_sec=300, user='root')
        if r.return_code != 0:
            raise RuntimeError('part upload of ' + remote + ' failed: ' + (r.stderr or '')[-300:])

    async def _create_sandbox(self):
        if self._is_verifier or not self._snapshot_id:
            return await super()._create_sandbox()
        if self._startup_env():
            raise RuntimeError("snapshot clones cannot take create-time env vars; pass them per command")
        kw = {}
        net = self._sandbox_create_network_options()
        if self.network_policy.network_mode != NetworkMode.PUBLIC or net.get("allow_out"):
            kw["network"] = net
        meta = {"environment_name": self.environment_name, "session_id": self.session_id,
                "owner": self._owner, "role": "clone", "snapshot_id": self._snapshot_id}
        if self._job:
            meta["asr_job"] = self._job
        self._sandbox = await AsyncSandbox.create(
            template=self._snapshot_id, metadata=meta, timeout=SANDBOX_TIMEOUT_SEC, allow_internet_access=True, **kw)
        self._write_midrun_receipt()

    def _write_midrun_receipt(self):
        """Tell `asr launch`'s mid-run snapshotter which sandbox to watch (a clone whose id nobody else records).

        The trial dir (not the sandbox) is what the launcher can see, so the receipt is the only handle on the
        live clone; it is written before the agent starts so no early state can be missed.
        """
        import json as _json
        import time as _time
        target = self.trial_paths.trial_dir / 'infra-validation'
        target.mkdir(parents=True, exist_ok=True)
        rec = {"sandbox_id": self._sandbox.sandbox_id, "session_id": self.session_id,
               "environment_name": self.environment_name, "snapshot_id": self._snapshot_id,
               "job": self._job, "ts": _time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (target / 'midrun-sandbox.json').write_text(_json.dumps(rec, indent=1) + "\n")
