import os
from pathlib import Path
import tempfile
import time
import unittest
from agent_snapshot_resume.replay import replay_inside as replay

class ReplayTimeout(unittest.TestCase):
    def test_timeout_header_at_start_of_long_output(self):
        self.assertTrue(hasattr(replay,'original_timeout'))
        self.assertEqual(replay.original_timeout('Exit code 143\nCommand timed out after 2m 0s\n'+'science\n'*100),120)
        self.assertIsNone(replay.original_timeout('science says Command timed out after 2m 0s'))
    def test_timeout_kills_entire_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'escaped'
            result=replay.run_bash(f"printf 'start\\n'; (sleep 0.5; touch '{f}') & wait",td,0.1)
            self.assertTrue(result['timeout']);self.assertIn('start',result['stdout'])
            time.sleep(0.6)
            self.assertFalse(f.exists(),'descendant survived replay timeout')

if __name__=='__main__':unittest.main()
