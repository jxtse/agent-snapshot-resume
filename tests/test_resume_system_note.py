import copy
import unittest
from agent_snapshot_resume.proxy import resume_proxy as proxy

class ResumeSystemNoteTests(unittest.TestCase):
    def test_system_note_between_result_and_resume_marker(self):
        body = {'messages': [
            {'role': 'assistant', 'content': [{'type':'tool_use','id':'tool1','name':'Bash','input':{'command':'true'}}]},
            {'role': 'user', 'content': [{'type':'tool_result','tool_use_id':'tool1','content':'real result'}, {'type':'text','text':proxy.CONTINUE_TXT}]},
            {'role':'system','content':'historical task reminder'},
            {'role':'assistant','content':proxy.SYNTH_TXT},
            {'role':'user','content':'UNIQUE_SENTINEL'},
            {'role':'assistant','content':'later genuine answer'},
        ]}
        spec={'sentinel':'UNIQUE_SENTINEL','mode':'hint','hint':'frozen hint'}
        out, acts = proxy.rewrite(copy.deepcopy(body),spec)
        self.assertIn('drop_continue_block',acts)
        self.assertEqual(out['messages'][1]['content'][0],body['messages'][1]['content'][0])
        self.assertEqual(out['messages'][2],body['messages'][2])
        self.assertEqual(out['messages'][-1],body['messages'][-1])
        self.assertEqual(sum(b.get('text','') == '<system-reminder>\nfrozen hint\n</system-reminder>' for m in out['messages'] for b in proxy._blocks(m)),1)
        self.assertNotIn(proxy.CONTINUE_TXT,str(out))

if __name__=='__main__': unittest.main()
