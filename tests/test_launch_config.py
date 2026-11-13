"""Job config generation: valid for Harbor, never contains a credential, fail-closed on ungated snapshots."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.bundle import build
from agent_snapshot_resume.launch import build_job_config, validate_job_config

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"


class LaunchConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.adapter = get_adapter("claude-code")
        self.bdir, self.meta = build(adapter=self.adapter, trial_dir=TRIAL, atif_step=4, out=self.root, task_dir=TASK)
        self.state = self.root / "snapshot-x.json"
        self.state.write_text(json.dumps({"snapshot_id": "snap-123", "replay_gate_passed": True,
                                          "fingerprint_source": "files=2\nabc", "fingerprint_command": "python3 -c x",
                                          "auxiliary_hashes": {}, "run_id": self.meta["run_id"]}))

    def cfg(self, **kw):
        args = dict(adapter=self.adapter, bundle=self.bdir, task_dir=TASK, job_name="j", jobs_dir=self.root / "jobs",
                    snapshot_state=self.state, fake=True, original_trial_dir=str(TRIAL))
        args.update(kw)
        return build_job_config(**args)

    def test_freeze_first_request_required_for_strict_real_runs(self):
        from agent_snapshot_resume.proxy.resume_proxy import request_digest
        meta = dict(json.loads((self.bdir / "bundle.json").read_text()),
                    orig_request_at_k={"req_bytes": 1, "content": {"message_block_sha256": []}})
        (self.bdir / "bundle.json").write_text(json.dumps(meta))
        route = ["--drop-fields", "metadata", "--strip-is-error"]
        trial = self.root / "fakejob" / "t__x" / "agent"
        logs = trial / "resume-proxy" / self.meta["run_id"]
        logs.mkdir(parents=True)
        (trial / "resume").mkdir()
        (trial / "resume" / "bundle.json").write_text(json.dumps(meta))
        body = {"model": "m", "messages": [], "temperature": 1}
        freeze = str(logs / "first_request_rewritten.json")
        (logs / "first_request_rewritten.json").write_text(json.dumps({"body": body}))
        good = {"ok": True, "request_sha256": request_digest(body), "upstream_fake": True,
                "route": {"drop_fields": ["metadata"], "strip_is_error": True, "strip_thinking": False}}
        kw = dict(fake=False, parity_gate="require", passthrough_args=route)
        with patch.dict(os.environ, {"ASR_UPSTREAM_URL": "https://example.invalid"}):
            with self.assertRaises(SystemExit):
                self.cfg(**kw)
            for bad in ({"ok": False}, {"request_sha256": "x"}, {"upstream_fake": False},
                        {"route": {"drop_fields": ["temperature"], "strip_is_error": True, "strip_thinking": False}}):
                (logs / "first_request_parity.json").write_text(json.dumps(dict(good, **bad)))
                with self.assertRaises(SystemExit, msg=str(bad)):
                    self.cfg(freeze_first_request=freeze, **kw)
            (logs / "first_request_parity.json").write_text(json.dumps(good))
            (trial / "resume" / "bundle.json").write_text(json.dumps(dict(meta, hint="other")))
            with self.assertRaises(SystemExit):
                self.cfg(freeze_first_request=freeze, **kw)
            (trial / "resume" / "bundle.json").write_text(json.dumps(meta))
            cfg = self.cfg(freeze_first_request=freeze, **kw)
            self.assertEqual(cfg["agents"][0]["kwargs"]["expected_first_request_sha256"], request_digest(body))

    def test_fake_config_valid_and_secret_free(self):
        secret = "test-secret-should-never-appear"
        with patch.dict(os.environ, {"ASR_UPSTREAM_KEY": secret, "ANTHROPIC_API_KEY": secret}):
            cfg = self.cfg()
        validate_job_config(cfg)
        text = json.dumps(cfg)
        self.assertNotIn(secret, text)
        self.assertNotIn("${ANTHROPIC_API_KEY}", text)
        agent = cfg["agents"][0]
        self.assertEqual(agent["import_path"], "agent_snapshot_resume.agents.claude_code:ClaudeCodeSnapshotResume")
        self.assertEqual(cfg["environment"]["import_path"],
                         "agent_snapshot_resume.sandbox.snapshot_env:SnapshotCloneEnvironment")
        self.assertEqual(agent["override_timeout_sec"], self.meta["agent_budget_s"])
        self.assertEqual(agent["kwargs"]["version"], "2.1.207")
        self.assertTrue(agent["kwargs"]["upstream_fake"])
        self.assertNotIn("ANTHROPIC_BASE_URL", agent["env"])
        self.assertEqual(agent["env"]["API_TIMEOUT_MS"], "3600000")
        env = cfg["environment"]["kwargs"]
        self.assertEqual(env["snapshot_id"], "snap-123")
        self.assertEqual(env["expected_fingerprint"], "files=2\nabc")
        self.assertIn("/tests/test.sh", env["expected_verifier_hashes"])

    def test_real_run_requires_upstream_and_allows_its_host(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASR_UPSTREAM_URL", None)
            with self.assertRaises(SystemExit):
                self.cfg(fake=False)
        cfg = self.cfg(fake=False, upstream_url="https://api.example.com/anthropic")
        validate_job_config(cfg)
        self.assertEqual(cfg["agents"][0]["extra_allowed_hosts"], ["api.example.com"])
        self.assertEqual(cfg["agents"][0]["kwargs"]["upstream_url"], "https://api.example.com/anthropic")

    def test_ungated_snapshot_refused(self):
        self.state.write_text(json.dumps({"snapshot_id": "snap-123", "replay_gate_passed": False}))
        with self.assertRaises(SystemExit):
            self.cfg()

    def test_snapshot_from_other_bundle_refused(self):
        d = json.loads(self.state.read_text())
        d["run_id"] = "other"
        self.state.write_text(json.dumps(d))
        with self.assertRaises(SystemExit):
            self.cfg()

    def test_templated_env_resolved_only_when_non_secret_and_present(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}):
            from agent_snapshot_resume.launch import original_agent_settings
            import shutil
            trial = self.root / "trial"
            shutil.copytree(TRIAL, trial)
            c = json.loads((trial / "config.json").read_text())
            c["agent"]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "${CLAUDE_CODE_MAX_OUTPUT_TOKENS}"
            (trial / "config.json").write_text(json.dumps(c))
            env, _ = original_agent_settings(str(trial))
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "64000")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_BASE_URL", env)

    def test_no_verifier(self):
        self.assertEqual(self.cfg(verifier=False)["verifier"], {"disable": True})

    def test_cli_dry_run_writes_and_prints(self):
        from agent_snapshot_resume.cli import main
        rc = main(["--env-file", "/nonexistent", "launch", "--bundle", str(self.bdir), "--task-dir", str(TASK),
                   "--snapshot-state", str(self.state), "--fake", "--no-verifier", "--dry-run",
                   "--jobs-dir", str(self.root / "jobs"), "--runs-dir", str(self.root / "runs")])
        self.assertEqual(rc, 0)
        files = list((self.root / "runs").rglob("job.json"))
        self.assertEqual(len(files), 1)
        validate_job_config(json.loads(files[0].read_text()))
