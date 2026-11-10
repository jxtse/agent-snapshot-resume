import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from agent_snapshot_resume.sandbox.snapshot_env import SnapshotCloneEnvironment

class EnvironmentGates(unittest.IsolatedAsyncioTestCase):
    def env(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        e=object.__new__(SnapshotCloneEnvironment);e.session_id='test__env'
        e.trial_paths=SimpleNamespace(trial_dir=Path(self.tmp.name));e._sandbox=SimpleNamespace(sandbox_id='fixture')
        return e
    async def test_auxiliary_mismatch_fails(self):
        e=self.env();e.exec=AsyncMock(return_value=SimpleNamespace(return_code=0,stdout=json.dumps({'/tmp/out':'wrong'})))
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):await e._check_runtime_hashes({'/tmp/out':'expected'},'auxiliary')
    async def test_runtime_hash_collection_failure_fails(self):
        e=self.env();e.exec=AsyncMock(return_value=SimpleNamespace(return_code=1,stdout='',stderr='missing file'))
        with self.assertRaises(RuntimeError):await e._check_runtime_hashes({'/tests/test.sh':'sha'},'verifier')
    async def test_verifier_exact_hash_records_success(self):
        e=self.env();e.exec=AsyncMock(return_value=SimpleNamespace(return_code=0,stdout=json.dumps({'/tests/test.sh':'sha'})))
        await e._check_runtime_hashes({'/tests/test.sh':'sha'},'verifier')
        p=e.trial_paths.trial_dir/'infra-validation/test__env-verifier.json'
        d=json.loads(p.read_text());self.assertEqual(d['expected'],d['actual'])
    async def test_clone_fingerprint_mismatch_aborts_before_agent(self):
        e=self.env();e._is_verifier=False;e._snapshot_id='snapshot';e._expected_fingerprint='expected';e._expected_auxiliary_hashes={}
        e._create_sandbox=AsyncMock();e.ensure_dirs=AsyncMock();e._mount_targets=lambda **kw:[]
        e.exec=AsyncMock(return_value=SimpleNamespace(return_code=0,stdout='wrong'))
        with self.assertRaisesRegex(RuntimeError,'fingerprint'):await e.start(False)

if __name__=='__main__':unittest.main()
