"""A hint longer than the 300-char extra-block preview must still be counted (exactly once)."""
import json
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.bundle import build
from agent_snapshot_resume.cli import main

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"


class LongHint(unittest.TestCase):
    def test_long_hint_counted(self):
        hint = ("Re-check every intermediate result before you write the final answer. " * 8).strip()
        self.assertGreater(len(hint), 400)
        with tempfile.TemporaryDirectory() as d:
            bdir, _ = build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=Path(d),
                            task_dir=TASK, mode="hint", hint=hint)
            self.assertEqual(main(["--env-file", "/nonexistent", "selfcheck", "--bundle", str(bdir)]), 0)
            rep = json.loads((bdir / "selfcheck" / "equivalence_report.json").read_text())
            self.assertEqual(rep["hint_blocks"], 1)
            self.assertTrue(rep["equivalent"])
