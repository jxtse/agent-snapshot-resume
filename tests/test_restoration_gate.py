import unittest
from agent_snapshot_resume.replay.gate import validate_replay, check_hashes

class GateTests(unittest.TestCase):
    def test_numeric_mismatch_rejected(self):
        a={'call_id':'a','side_effect':'exec','is_error':False}
        for fidelity in ('numbers_differ','differs','timeout','replayer_exception'):
            with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[{'call_id':'a','fidelity':fidelity,'rc':0}]},'0')
    def test_changed_exit_rejected(self):
        a={'call_id':'a','side_effect':'exec','is_error':True}
        with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[{'call_id':'a','fidelity':'exact','rc':0}]},'0')
    def test_background_file_required(self):
        a={'call_id':'a','side_effect':'exec_background','bg_status':'completed','background_output_file':'/tmp/x'}
        d={'call_id':'a','fidelity':'bg_rerun_foreground','rc':0}
        with self.assertRaises(RuntimeError):validate_replay({'actions':[a]},{'results':[d]},'0')
        d.update(background_output_file='/tmp/x',background_output_sha256='hash')
        self.assertTrue(validate_replay({'actions':[a]},{'results':[d]},'0'))
    def test_missing_hash_and_mismatch_fail(self):
        with self.assertRaises(RuntimeError):check_hashes({'x':'sha'}, {}, 'fixture')
        with self.assertRaises(RuntimeError):check_hashes({'x':'sha'}, {'x':'wrong'}, 'fixture')
        self.assertTrue(check_hashes({'x':'sha'}, {'x':'sha'}, 'fixture'))

if __name__=='__main__':unittest.main()
