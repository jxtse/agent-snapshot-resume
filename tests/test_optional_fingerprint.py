import subprocess
import sys
import shlex
from pathlib import Path
import tempfile
import unittest
from agent_snapshot_resume.replay.fingerprint import build_fingerprint_command

class OptionalFingerprint(unittest.TestCase):
    def test_missing_optional_root_is_bound_not_fabricated(self):
        with tempfile.TemporaryDirectory() as td:
            missing=str(Path(td)/'not-produced.json')
            cmd=build_fingerprint_command([td,missing],optional_roots=[missing]).replace('python3 -c',shlex.quote(sys.executable)+' -c',1)
            before=subprocess.run(['bash','-c',cmd],capture_output=True,text=True)
            self.assertEqual(before.returncode,0,before.stderr)
            Path(missing).write_text('{}')
            after=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,check=True)
            self.assertNotEqual(before.stdout,after.stdout)
    def test_required_root_still_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r=subprocess.run(['bash','-c',build_fingerprint_command([str(Path(td)/'required')])],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
if __name__=='__main__':unittest.main()
