"""finish_prepared_source must never overwrite recorded evidence or snapshot after the task files changed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_snapshot_resume.sandbox import source as mod


class CompletionObserver(unittest.TestCase):
    def test_changed_pending_fingerprint_cannot_be_overwritten_or_snapshotted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / 'bundle'
            bundle.mkdir()
            state = root / 'snapshot-test.json'
            plan = b'{"actions": []}'
            (bundle / 'plan.json').write_bytes(plan)
            (bundle / 'bundle.json').write_text(json.dumps({'fingerprint_roots': ['/app']}))
            state.write_text(json.dumps({'bundle': str(bundle), 'tag': 'test', 'owner': 'asr',
                                         'source_sandbox_id': 'id', 'replay_pid': 9,
                                         'pending_audit_fingerprint': 'files=1\nORIGINAL'}))
            sandbox = MagicMock()
            sandbox.sandbox_id = 'id'
            sandbox.get_info.return_value.metadata = {'owner': 'asr', 'role': 'source', 'asr_run': 'test'}
            files = {f'{mod.REPLAY_DIR}/plan.json': plan,
                     f'{mod.REPLAY_DIR}/replay_inside.py': mod.REPLAY_SCRIPT.read_bytes(),
                     f'{mod.REPLAY_DIR}/replay_report.json': b'{"results": []}',
                     f'{mod.REPLAY_DIR}/replay_stdout.log': b''}
            sandbox.files.read.side_effect = lambda name, **kw: files[name]
            sandbox.commands.run.side_effect = lambda command, **kw: MagicMock(
                stdout='0' if command.startswith('cat ') else 'files=1\nCHANGED')
            fake_cls = MagicMock()
            fake_cls.connect.return_value = sandbox
            with patch.object(mod, '_sandbox_cls', return_value=fake_cls):
                with self.assertRaisesRegex(RuntimeError, 'fingerprint changed'):
                    mod.finish_prepared_source(state)
            sandbox.create_snapshot.assert_not_called()
            sandbox.kill.assert_not_called()
            self.assertEqual(json.loads(state.read_text())['pending_audit_fingerprint'], 'files=1\nORIGINAL')

    def test_already_dispatched_snapshot_refused(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / 's.json'
            state.write_text(json.dumps({'snapshot_dispatch_started': True}))
            with self.assertRaisesRegex(RuntimeError, 'already dispatched'):
                mod.finish_prepared_source(state)
