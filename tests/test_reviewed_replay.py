import hashlib
import unittest
from agent_snapshot_resume.replay.gate import validate_replay

class ReviewedReplay(unittest.TestCase):
    def check(self,kind,old,out,err,rc,orig_error=False,extra=None,command='python x.py | grep value'):
        a={'call_id':'x','tool':'Bash','side_effect':'exec','is_error':orig_error,'args':{'command':command},'orig_text':old}
        r={'call_id':'x','fidelity':'timeout_reproduced' if orig_error else 'differs','rc':rc,'stdout_full':out,'stderr_full':err,**(extra or {})}
        review={'x':{'kind':kind,'command_sha256':hashlib.sha256(command.encode()).hexdigest()}}
        return validate_replay({'actions':[a]},{'results':[r]},0,review)
    def test_terminal_grep_only_clock_stderr_allowed(self):
        self.assertTrue(self.check('terminal_grep_with_shell_time','fixed\nreal\t0m0.009s\nuser\t0m0.01s\nsys\t0m0s','fixed\n','real\t0m0.008s\nuser\t0m0.006s\nsys\t0m0s',1))
        with self.assertRaises(RuntimeError):self.check('terminal_grep_with_shell_time','fixed\nreal\t0m0.009s','changed\n','real\t0m0.008s',1)
        with self.assertRaises(RuntimeError):self.check('terminal_grep_with_shell_time','fixed\nreal\t0m0.009s','fixed\n','ImportError\nreal\t0m0.008s',1)
    def test_original_timeout_requires_same_deadline_and_science(self):
        old='Exit code 143\nCommand timed out after 2m 0s\nP=1.5\nreal 19.11'
        extra={'orig_timed_out':True,'timeout_s':120,'status':'timeout'}
        self.assertTrue(self.check('original_timeout_partial_output',old,'P=1.5\nreal 20.1','',None,True,extra))
        for out,data in [('P=1.7\nreal 20.1',extra),('P=1.5\nreal 20.1',dict(extra,timeout_s=270)),('P=1.5\nreal 20.1',dict(extra,orig_timed_out=False))]:
            with self.assertRaises(RuntimeError):self.check('original_timeout_partial_output',old,out,'',None,True,data)

if __name__=='__main__':unittest.main()
