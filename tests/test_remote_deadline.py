import inspect
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from agent_snapshot_resume.agents import base as harness
from agent_snapshot_resume.agents import claude_code


class RemoteDeadline(unittest.TestCase):
    def wrap(self, command, seconds):
        self.assertTrue(hasattr(harness, 'bounded_remote_command'),
                        'resume launch needs an in-sandbox deadline independent of the host controller')
        return harness.bounded_remote_command(command, seconds)

    def test_remote_timeout_stops_child_and_keeps_output(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / 'should-not-exist'
            cmd = "printf 'before\\n'; (sleep 0.6; touch '" + str(marker) + "') & wait"
            r = subprocess.run(['bash', '-c', self.wrap(cmd, 0.1)], capture_output=True, text=True, timeout=4)
            self.assertEqual(r.returncode, 124, r.stderr)
            self.assertEqual(r.stdout, 'before\n')
            time.sleep(0.7)
            self.assertFalse(marker.exists(), 'grandchild escaped the remote deadline')

    def test_pipe_failure_is_not_masked_by_tee(self):
        r = subprocess.run(['bash', '-c', self.wrap("printf 'data\\n'; (exit 7) | tee /dev/null", 2)],
                           capture_output=True, text=True, timeout=4)
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertEqual(r.stdout, 'data\n')

    def test_invalid_budget_rejected(self):
        for bad in [0, -1, float('inf'), float('nan')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.wrap('true', bad)

    def test_production_run_uses_bound_from_frozen_bundle(self):
        source = inspect.getsource(claude_code.ClaudeCodeSnapshotResume.run)
        self.assertIn('bounded_remote_command(', source)
        self.assertIn("meta['agent_budget_s']", source)


if __name__ == '__main__':
    unittest.main()
