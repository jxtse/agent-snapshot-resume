import hashlib
import unittest
from agent_snapshot_resume.replay.gate import validate_replay

CMD="python3 - <<'PY'\nimport cProfile,pstats,io\npr=cProfile.Profile(); pr.enable()\nprint('P=',run())\npr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(2)\nPY"
OLD='''P= 1.3635581310617684
         100 function calls (90 primitive calls) in 93.188 seconds

   Ordered by: cumulative time
   List reduced from 14 to 2 due to restriction <2>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004   93.188   93.188 /app/model/fit.py:984(analyze)
        2   44.730   22.365   77.873   38.937 /app/model/fit.py:227(search)
Shell cwd was reset to /app'''
NEW=OLD.replace('93.188','94.188').replace('44.730','45.730').replace('Shell cwd was reset to /app','')


class ProfileReplayGate(unittest.TestCase):
    def check(self,text,review=True,cmd=CMD):
        action={'call_id':'profile','tool':'Bash','side_effect':'exec','is_error':False,'args':{'command':cmd},'orig_text':OLD}
        result={'call_id':'profile','fidelity':'numbers_differ','rc':0,'stdout_full':text,'stderr_full':''}
        reviews={'profile':{'kind':'cprofile_timings','command_sha256':hashlib.sha256(CMD.encode()).hexdigest()}} if review else {}
        return validate_replay({'actions':[action]},{'results':[result]},0,reviews)

    def test_exact_reviewed_aggregate_pair_only(self):
        text=NEW.replace('100 function calls (90 primitive calls)','93 function calls (83 primitive calls)')
        action={'call_id':'profile','tool':'Bash','side_effect':'exec','is_error':False,'args':{'command':CMD},'orig_text':OLD}
        result={'call_id':'profile','fidelity':'numbers_differ','rc':0,'stdout_full':text,'stderr_full':''}
        reviews={'profile':{'kind':'cprofile_reviewed_aggregate_counts','command_sha256':hashlib.sha256(CMD.encode()).hexdigest(),
                            'original_aggregate_counts':['100','90'],'replayed_aggregate_counts':['93','83']}}
        self.assertTrue(validate_replay({'actions':[action]},{'results':[result]},0,reviews))
        with self.assertRaises(RuntimeError):
            validate_replay({'actions':[action]},{'results':[dict(result,stdout_full=text.replace('93 function calls','94 function calls'))]},0,reviews)
    def test_known_performance_columns_only(self):self.assertTrue(self.check(NEW))
    def test_science_change_fails(self):
        with self.assertRaises(RuntimeError):self.check(NEW.replace('1.3635581310617684','2.0'))
    def test_call_counts_change_fails(self):
        with self.assertRaises(RuntimeError):self.check(NEW.replace('100 function calls','101 function calls'))
    def test_unreviewed_or_different_command_fails(self):
        with self.assertRaises(RuntimeError):self.check(NEW,review=False)
        with self.assertRaises(RuntimeError):self.check(NEW,cmd=CMD+'\nprintf extra')
    def test_unparsed_output_fails(self):
        with self.assertRaises(RuntimeError):self.check(NEW+'\nscience=42')
    def test_profiled_function_or_counts_changed_fails(self):
        with self.assertRaises(RuntimeError):self.check(NEW.replace('984(analyze)','984(other)'))
        with self.assertRaises(RuntimeError):self.check(NEW.replace('        2   45.730','        3   45.730'))


if __name__=='__main__':unittest.main()
