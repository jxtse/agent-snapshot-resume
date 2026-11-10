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
"""
import json
import shlex

from e2b import AsyncSandbox
from harbor.models.task.config import NetworkMode

from ..config import owner_tag
from ..replay.fingerprint import DEFAULT_ROOTS, build_fingerprint_command
from ..replay.gate import check_hashes
from .command_recovery import SnapshotCommandRecovery
from .fc_env import SANDBOX_TIMEOUT_SEC, SandboxEnvironment

DEFAULT_FINGERPRINT = build_fingerprint_command(list(DEFAULT_ROOTS))


class SnapshotCloneEnvironment(SnapshotCommandRecovery, SandboxEnvironment):
    def __init__(self, *args, snapshot_id: str | None = None, expected_fingerprint=None,
                 expected_auxiliary_hashes=None, expected_verifier_hashes=None, fingerprint_command=None,
                 owner: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_id = snapshot_id
        self._expected_fingerprint = expected_fingerprint
        self._fingerprint_command = fingerprint_command or DEFAULT_FINGERPRINT
        self._expected_auxiliary_hashes = expected_auxiliary_hashes or {}
        self._expected_verifier_hashes = expected_verifier_hashes or {}
        self._owner = owner or owner_tag()
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
        code = ('import hashlib,json,pathlib; paths=json.loads(' + repr(json.dumps(list(expected))) + '); '
                'print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))')
        result = await self.exec('python3 -c ' + shlex.quote(code), timeout_sec=120, user='root')
        if result.return_code != 0:
            raise RuntimeError(label + ' runtime file-hash collection failed: ' + (result.stderr or '')[-1500:])
        actual = json.loads(result.stdout)
        record = {'sandbox_id': self._sandbox.sandbox_id, 'label': label, 'expected': expected, 'actual': actual}
        target = self.trial_paths.trial_dir / 'infra-validation'
        target.mkdir(parents=True, exist_ok=True)
        (target / (self.session_id + '-' + label + '.json')).write_text(json.dumps(record, indent=2))
        check_hashes(expected, actual, label)

    async def _create_sandbox(self):
        if self._is_verifier or not self._snapshot_id:
            return await super()._create_sandbox()
        if self._startup_env():
            raise RuntimeError("snapshot clones cannot take create-time env vars; pass them per command")
        kw = {}
        net = self._sandbox_create_network_options()
        if self.network_policy.network_mode != NetworkMode.PUBLIC or net.get("allow_out"):
            kw["network"] = net
        self._sandbox = await AsyncSandbox.create(
            template=self._snapshot_id,
            metadata={"environment_name": self.environment_name, "session_id": self.session_id,
                      "owner": self._owner, "role": "clone", "snapshot_id": self._snapshot_id},
            timeout=SANDBOX_TIMEOUT_SEC, allow_internet_access=True, **kw)
