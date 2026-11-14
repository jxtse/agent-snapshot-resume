"""A snapshot state that predates the optional-fingerprint-roots field must still be shareable.

The cut identity gained `fingerprint_optional_roots` after some snapshots were already prepared (state files record
null there). Refusing those states blocks every launch of an otherwise valid snapshot, so the comparison treats an
absent/empty optional-root list as identical to a bundle that names none - but only when the state's own fingerprint
command shows that it fingerprinted with no optional roots, and never in the direction that would weaken the gate.
"""
import json
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.bundle import build, cut_identity_diff, snapshot_cut_identity
from agent_snapshot_resume.launch import build_job_config
from agent_snapshot_resume.replay.fingerprint import build_fingerprint_command

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"


def _rewrite_meta(bdir: Path, **fields) -> None:
    meta = json.loads((bdir / "bundle.json").read_text())
    meta.update(fields)
    (bdir / "bundle.json").write_text(json.dumps(meta, indent=1))


class OptionalRootsIdentityCompat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.adapter = get_adapter("claude-code")
        self.bdir, self.meta = build(adapter=self.adapter, trial_dir=TRIAL, atif_step=4, out=self.root, task_dir=TASK)
        self.state = self.root / "snapshot-x.json"

    def write_state(self, cut_identity, fingerprint_command=None):
        self.state.write_text(json.dumps({
            "snapshot_id": "snap-123", "replay_gate_passed": True, "fingerprint_source": "files=2\nabc",
            "fingerprint_command": fingerprint_command or build_fingerprint_command(["/app"]),
            "auxiliary_hashes": {}, "run_id": self.meta["run_id"], "agent_version": self.meta["agent_version_orig"],
            "cut_identity": cut_identity}))

    def launch(self):
        return build_job_config(adapter=self.adapter, bundle=self.bdir, task_dir=TASK, job_name="j",
                                jobs_dir=self.root / "jobs", snapshot_state=self.state, fake=True,
                                original_trial_dir=str(TRIAL))

    def old_identity(self, key="fingerprint_optional_roots", value=None, drop=False):
        ident = snapshot_cut_identity(self.bdir)
        if drop:
            ident.pop(key, None)
        else:
            ident[key] = value
        return ident

    def test_null_optional_roots_accepted_and_rebound(self):
        old = self.old_identity()
        self.assertIsNone(old["fingerprint_optional_roots"])
        self.write_state(old)
        self.launch()
        rec = json.loads(self.state.read_text())
        self.assertEqual(rec["cut_identity"], snapshot_cut_identity(self.bdir))  # rebound to the current form
        self.assertEqual(rec["cut_identity_rebound"]["from"]["fingerprint_optional_roots"], None)

    def test_missing_key_accepted(self):
        self.write_state(self.old_identity(drop=True))
        self.launch()
        self.assertNotIn("fingerprint_optional_roots", json.loads(self.state.read_text())["cut_identity_rebound"]["from"])

    def test_command_with_optional_roots_refused(self):
        """The state's own fingerprint command shows a tolerated missing root: not the same fingerprint as ours."""
        self.write_state(self.old_identity(),
                         fingerprint_command=build_fingerprint_command(["/app", "/app/maybe"], ["/app/maybe"]))
        with self.assertRaises(SystemExit) as ctx:
            self.launch()
        self.assertIn("fingerprint_optional_roots", str(ctx.exception))
        self.assertNotIn("cut_identity_rebound", json.loads(self.state.read_text()))  # refused: nothing rewritten

    def test_bundle_adding_optional_roots_refused(self):
        _rewrite_meta(self.bdir, fingerprint_roots=["/app", "/app/maybe"], fingerprint_optional_roots=["/app/maybe"])
        self.write_state(self.old_identity())
        with self.assertRaises(SystemExit) as ctx:
            self.launch()
        self.assertIn("fingerprint_optional_roots", str(ctx.exception))

    def test_other_identity_fields_still_enforced(self):
        old = self.old_identity()
        old["plan_sha256"] = "0" * 64
        self.write_state(old)
        with self.assertRaises(SystemExit) as ctx:
            self.launch()
        self.assertIn("plan_sha256", str(ctx.exception))

    def test_diff_helper_is_one_directional(self):
        mine = {"fingerprint_optional_roots": [], "cut_line": 47}
        self.assertEqual(cut_identity_diff(mine, {"fingerprint_optional_roots": None, "cut_line": 47}), [])
        self.assertEqual(cut_identity_diff(dict(mine, fingerprint_optional_roots=["/app/x"]),
                                           {"fingerprint_optional_roots": None, "cut_line": 47}),
                         ["fingerprint_optional_roots"])
        self.assertEqual(cut_identity_diff(mine, {"fingerprint_optional_roots": "/app/x", "cut_line": 47}),
                         ["fingerprint_optional_roots"])


if __name__ == "__main__":
    unittest.main()
