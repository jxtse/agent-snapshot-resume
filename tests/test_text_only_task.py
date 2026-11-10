import copy
import hashlib
import unittest
from agent_snapshot_resume.replay.gate import validate_replay
from agent_snapshot_resume.replay.replay_inside import compare
from agent_snapshot_resume.proxy import resume_proxy as proxy

class TextOnlyTaskTests(unittest.TestCase):
    def test_text_only_rejects_nested_images_without_mutating(self):
        b={'messages':[{'role':'user','content':[{'type':'tool_result','tool_use_id':'t','content':[{'type':'image','source':{'type':'base64','data':'AA=='}}]}]}]}
        before=copy.deepcopy(b)
        with self.assertRaises(ValueError):proxy.rewrite(b,{'sentinel':'S','require_text_only':True})
        self.assertEqual(b,before)
    def test_text_only_allows_plain_text_without_sentinel(self):
        b={'messages':[{'role':'user','content':'plain data'}]}
        out,_=proxy.rewrite(copy.deepcopy(b),{'sentinel':'S','require_text_only':True})
        self.assertEqual(out,b)
    def test_shell_clock_normalized_not_science(self):
        self.assertEqual(compare('P=2.73\nreal\t1m3.1s','P=2.73\nreal\t0m59.8s',0),'volatile_only')
        self.assertNotEqual(compare('P=2.73\nreal\t1m3.1s','P=9.73\nreal\t0m59.8s',0),'volatile_only')
    def test_posix_shell_clock_normalized_not_science(self):
        self.assertEqual(compare('P=2.73\nreal 13.5\nuser 12.2\nsys 1.3',
                                 'P=2.73\nreal 12.5\nuser 11.2\nsys 1.3',0),'volatile_only')
        self.assertNotEqual(compare('P=2.73\nreal 13.5','P=9.73\nreal 12.5',0),'volatile_only')
    def test_hardware_probe_preserves_package_and_cpu_values(self):
        cmd='version; nproc; free -g';old='numpy 1.26.4\n2\n total used free shared buff/cache available\nMem: 251 0 250 0 0 250'
        a={'call_id':'x','side_effect':'exec','is_error':False,'args':{'command':cmd},'orig_text':old}
        r={'call_id':'x','fidelity':'numbers_differ','rc':0,'stdout_full':old.replace('251 0 250 0 0 250','3 0 2 0 0 2'),'stderr_full':''}
        reviews={'x':{'kind':'shell_memory_probe','command_sha256':hashlib.sha256(cmd.encode()).hexdigest()}}
        self.assertTrue(validate_replay({'actions':[a]},{'results':[r]},0,reviews))
        for invalid in [r['stdout_full'].replace('1.26.4','2.1.0'),r['stdout_full'].replace('\n2\n','\n8\n')]:
            with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[dict(r,stdout_full=invalid)]},0,reviews)
    def test_deferred_task_management_is_only_for_task_tools(self):
        a={'call_id':'x','side_effect':'task_mgmt','tool':'TaskCreate'};r={'call_id':'x','fidelity':'task_mgmt_restored_from_session'}
        self.assertTrue(validate_replay({'actions':[a]},{'results':[r]},0))
        with self.assertRaises(RuntimeError):validate_replay({'actions':[dict(a,tool='CronCreate')]},{'results':[r]},0)
    def test_reviewed_terminal_grep_preserves_stdout_and_command(self):
        cmd="python3 patch.py\npython3 absent.py 2>&1 | grep -E 'diagnostic'"
        a={'call_id':'x','side_effect':'exec','is_error':False,'args':{'command':cmd},'orig_text':'fixed'}
        r={'call_id':'x','fidelity':'differs','rc':1,'stdout_full':'fixed\n','stderr_full':''}
        reviews={'x':{'kind':'terminal_grep_no_match','command_sha256':hashlib.sha256(cmd.encode()).hexdigest()}}
        self.assertTrue(validate_replay({'actions':[a]},{'results':[r]},0,reviews))
        with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[dict(r,stdout_full='changed')]},0,reviews)
        with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[r]},0)

if __name__=='__main__':unittest.main()
