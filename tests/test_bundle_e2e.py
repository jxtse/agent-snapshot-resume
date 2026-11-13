"""Offline end-to-end: synthetic trial -> bundle -> simulated resumed request -> proxy rewrite -> equivalence."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume import equiv
from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.agents.claude_code import simulate_resume_request
from agent_snapshot_resume.bundle import build
from agent_snapshot_resume.proxy.resume_proxy import rewrite
from agent_snapshot_resume.replay import replay_inside

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"


class BundleE2E(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.bdir, self.meta = build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=self.out,
                                     task_dir=TASK)

    def session_lines(self):
        return (TRIAL / "agent/sessions/projects/-app/00000000-0000-4000-8000-000000000001.jsonl").read_text().splitlines()

    def test_truncation_is_byte_exact_prefix_ending_at_content(self):
        trunc = (self.bdir / "session_trunc.jsonl").read_text()
        orig = self.session_lines()
        self.assertEqual(trunc, "\n".join(orig[:self.meta["cut_line"]]) + "\n")
        # the trailing last-prompt line before step 4 is trimmed
        self.assertEqual(json.loads(trunc.splitlines()[-1])["message"]["content"][0]["tool_use_id"], "toolu_b1")

    def test_cut_line_selects_the_same_step(self):
        b2, m2 = build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=None, out=self.out / "c",
                       task_dir=TASK, cut_line=self.meta["cut_line"])
        self.assertEqual(m2["resume_atif_step"], 4)
        self.assertEqual((b2 / "session_trunc.jsonl").read_bytes(), (self.bdir / "session_trunc.jsonl").read_bytes())
        self.assertEqual(m2["n_compactions_before_cut"], 0)
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=None, out=self.out / "d",
                  task_dir=TASK, cut_line=1)  # not the start of an agent step
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=self.out / "e",
                  task_dir=TASK, cut_line=self.meta["cut_line"])

    def test_budget_and_metadata(self):
        self.assertEqual(self.meta["agent_timeout_task_s"], 600.0)
        self.assertAlmostEqual(self.meta["agent_budget_s"], 600.0 - self.meta["elapsed_at_k_s"], places=1)
        self.assertEqual(self.meta["model"], "test-model")
        self.assertEqual(self.meta["agent_version_orig"], "2.1.207")
        spec = json.loads((self.bdir / "proxy_spec.json").read_text())
        self.assertNotIn("key_file", spec)
        self.assertEqual(spec["pin_date"], "2026-01-01")

    def test_plan_replays_in_a_scratch_workdir(self):
        plan = json.loads((self.bdir / "plan.json").read_text())
        self.assertEqual([a["tool"] for a in plan["actions"]], ["Write", "Bash"])
        work = self.out / "app"
        work.mkdir()
        for a in plan["actions"]:  # retarget /app -> scratch dir
            a["cwd"] = str(work)
            if a["tool"] == "Write":
                a["args"]["file_path"] = str(work / "sum.py")
            else:
                a["args"]["command"] = a["args"]["command"].replace("/app", str(work))
        plan["initial_cwd"] = str(work)
        (self.out / "plan.json").write_text(json.dumps(plan))
        subprocess.run([sys.executable, replay_inside.__file__, str(self.out / "plan.json"),
                        str(self.out / "report.json")], check=True, capture_output=True)
        rep = json.loads((self.out / "report.json").read_text())
        self.assertEqual([r["fidelity"] for r in rep["results"]], ["exact", "exact"])

    def test_rewritten_request_is_equivalent_and_raw_is_not(self):
        spec = json.loads((self.bdir / "proxy_spec.json").read_text())
        lines = [json.loads(l) for l in (self.bdir / "session_trunc.jsonl").read_text().splitlines()]
        raw = simulate_resume_request(lines, spec["sentinel"])
        new, actions = rewrite(json.loads(json.dumps(raw)), spec)
        self.assertIn("drop_synthetic_assistant", actions)
        self.assertIn("drop_continue_block", actions)
        self.assertIn("pin_date", actions)
        for name, body in (("raw", raw), ("new", new)):
            (self.out / f"{name}.json").write_text(json.dumps(body))
        ok = equiv.check(self.bdir / "session_trunc.jsonl", self.meta["cut_line"], self.out / "new.json",
                         sentinel=spec["sentinel"])
        bad = equiv.check(self.bdir / "session_trunc.jsonl", self.meta["cut_line"], self.out / "raw.json",
                          sentinel=spec["sentinel"])
        self.assertTrue(ok["equivalent"], ok)
        self.assertFalse(bad["equivalent"])
        self.assertIn("Today's date is 2026-01-01.", json.dumps(new))

    def test_hint_mode_adds_exactly_one_block(self):
        hdir, meta = build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=self.out,
                           task_dir=TASK, mode="hint", hint="Check the output file.")
        spec = json.loads((hdir / "proxy_spec.json").read_text())
        lines = [json.loads(l) for l in (hdir / "session_trunc.jsonl").read_text().splitlines()]
        new, _ = rewrite(simulate_resume_request(lines, spec["sentinel"]), spec)
        (self.out / "h.json").write_text(json.dumps(new))
        r = equiv.check(hdir / "session_trunc.jsonl", meta["cut_line"], self.out / "h.json", hint=meta["hint"],
                        sentinel=spec["sentinel"])
        self.assertTrue(r["equivalent"], r)
        self.assertEqual(r["hint_blocks"], 1)

    def test_existing_bundle_requires_force(self):
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=self.out, task_dir=TASK)

    def test_non_agent_step_rejected(self):
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=1, out=self.out / "x", task_dir=TASK)

    def test_cli_selfcheck(self):
        from agent_snapshot_resume.cli import main
        self.assertEqual(main(["--env-file", "/nonexistent", "selfcheck", "--bundle", str(self.bdir)]), 0)
