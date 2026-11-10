"""Claude Code 2.1.280 pin, passthrough proxy extra flags, and snapshot sharing between bundles of one cut."""
import json
import shlex
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.agents.claude_code import CC_SHA256, ClaudeCodeSnapshotResumeOptions
from agent_snapshot_resume.bundle import build, snapshot_cut_identity
from agent_snapshot_resume.launch import build_job_config, validate_job_config
from agent_snapshot_resume.proxy import runtime

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"


class CC280(unittest.TestCase):
    def test_pin_matches_official_manifest(self):
        # https://downloads.claude.ai/claude-code-releases/2.1.280/manifest.json platforms.linux-x64.checksum
        self.assertEqual(CC_SHA256["2.1.280"], "1e08503dbdf3c2cb0d706d32f3408277388d1c76ef108673e8fe42c1b322925b")
        cmds = dict((n, c) for n, c, _ in get_adapter("claude-code").install_commands("2.1.280"))
        self.assertIn("claude-code-releases/2.1.280/linux-x64/claude", cmds["download_cc"])
        self.assertIn(CC_SHA256["2.1.280"] + "  /tmp/.asr-claude' | sha256sum -c -", cmds["download_cc"])


class PassthroughArgs(unittest.TestCase):
    def test_whitelist(self):
        self.assertEqual(runtime.validate_passthrough_args("--drop-fields metadata --strip-is-error"),
                         ["--drop-fields", "metadata", "--strip-is-error"])
        for bad in ("--upstream http://evil", "--drop-fields", "--drop-fields 'a;rm -rf /'", "--log-dir /tmp",
                    "--key-file /etc/passwd", "; echo"):
            with self.assertRaises(ValueError, msg=bad):
                runtime.validate_passthrough_args(bad)

    def test_start_command_carries_flags(self):
        extra = " ".join(shlex.quote(x) for x in runtime.validate_passthrough_args(
            ["--drop-fields", "metadata", "--strip-is-error"]))
        cmd = runtime.passthrough_start_command("https://up.example.com", extra=extra)
        self.assertIn("--cafile $P/ca-bundle.crt --drop-fields metadata --strip-is-error", cmd)

    def test_agent_option(self):
        o = ClaudeCodeSnapshotResumeOptions(passthrough_args=["--drop-fields", "metadata", "--strip-is-error"])
        self.assertEqual(o.passthrough_args, ["--drop-fields", "metadata", "--strip-is-error"])


class SharedSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.adapter = get_adapter("claude-code")
        self.same, self.meta = build(adapter=self.adapter, trial_dir=TRIAL, atif_step=4, out=self.root, task_dir=TASK)
        self.hint, self.hmeta = build(adapter=self.adapter, trial_dir=TRIAL, atif_step=4, out=self.root,
                                      task_dir=TASK, mode="hint", hint="Check your arithmetic.")
        self.other, _ = build(adapter=self.adapter, trial_dir=TRIAL, atif_step=3, out=self.root, task_dir=TASK)
        self.state = self.root / "snapshot-x.json"
        self.state.write_text(json.dumps({
            "snapshot_id": "snap-123", "replay_gate_passed": True, "fingerprint_source": "files=2\nabc",
            "fingerprint_command": "python3 -c x", "auxiliary_hashes": {}, "run_id": self.meta["run_id"],
            "bundle": str(self.same), "cut_identity": snapshot_cut_identity(self.same),
            "agent_version": self.meta["agent_version_orig"]}))

    def cfg(self, bundle, **kw):
        return build_job_config(adapter=self.adapter, bundle=bundle, task_dir=TASK, job_name="j",
                                jobs_dir=self.root / "jobs", snapshot_state=self.state, fake=True,
                                original_trial_dir=str(TRIAL), **kw)

    def test_same_cut_hint_bundle_may_share(self):
        self.assertNotEqual(self.meta["run_id"], self.hmeta["run_id"])
        self.assertEqual(snapshot_cut_identity(self.same), snapshot_cut_identity(self.hint))
        cfg = self.cfg(self.hint)
        validate_job_config(cfg)
        self.assertEqual(cfg["environment"]["kwargs"]["snapshot_id"], "snap-123")
        self.assertTrue(cfg["agents"][0]["kwargs"]["resume_bundle"].endswith(self.hmeta["run_id"]))

    def test_other_cut_refused(self):
        with self.assertRaises(SystemExit):
            self.cfg(self.other)

    def test_changed_plan_refused(self):
        p = self.hint / "plan.json"
        plan = json.loads(p.read_text())
        plan["actions"] = plan["actions"][:-1]
        p.write_text(json.dumps(plan))
        with self.assertRaises(SystemExit):
            self.cfg(self.hint)

    def test_passthrough_args_in_job_config(self):
        cfg = self.cfg(self.same, passthrough_args=["--drop-fields", "metadata", "--strip-is-error"])
        validate_job_config(cfg)
        self.assertEqual(cfg["agents"][0]["kwargs"]["passthrough_args"],
                         ["--drop-fields", "metadata", "--strip-is-error"])
        with self.assertRaises(ValueError):
            self.cfg(self.same, passthrough_args=["--upstream", "http://x"])
