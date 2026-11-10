import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from agent_snapshot_resume.replay import replay_inside as r

class ReplayTests(unittest.TestCase):
    def test_reordering_with_addresses_preserves_numbers(self):
        self.assertEqual(r.compare('Exit code 1\nWarning at 0xabcdef1234\nvalue 1.386','value 1.386\nWarning at 0x1234abcdef',1),'address_and_order_only')
        self.assertNotIn(r.compare('Warning at 0xabcdef1234\nvalue 1.386','value 9.386\nWarning at 0x1234abcdef',0),('exact','normalized','address_and_order_only'))
    def test_warning_reorder_plus_header_alignment(self):
        self.assertEqual(r.compare('warning\n    n  value\nC01  1.234','n  value\nC01  1.234\n\nwarning',0),'line_alignment_and_order_only')
        self.assertNotEqual(r.compare('warning\n    n  value\nC01  1.234','n  value\nC01  9.234\n\nwarning',0),'line_alignment_and_order_only')

    def test_background_file_restored_with_actual_stdout(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);out=p/'tasks/job.output'
            plan={'initial_cwd':d,'actions':[{'tool':'Bash','side_effect':'exec_background','args':{'command':'printf actual-background-output'},'call_id':'test123','background_output_file':str(out),'bg_finished_before_k':True,'bg_status':'completed','is_error':False,'cwd':d}]}
            (p/'plan.json').write_text(json.dumps(plan))
            subprocess.run([sys.executable,r.__file__,str(p/'plan.json'),str(p/'report.json')],check=True,capture_output=True)
            self.assertEqual(out.read_text(),'actual-background-output')
            result=json.loads((p/'report.json').read_text())['results'][0]
            self.assertEqual(result['stdout_full'],'actual-background-output')
            self.assertTrue(result['background_output_sha256'])

if __name__=='__main__':unittest.main()
