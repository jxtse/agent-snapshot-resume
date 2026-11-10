import unittest
from unittest.mock import MagicMock
from agent_snapshot_resume.sandbox import source as p


class SourceDispatch(unittest.TestCase):
    def test_sdk_background_not_shell_ampersand(self):
        self.assertTrue(hasattr(p, 'dispatch_replay'))
        sandbox=MagicMock()
        sandbox.commands.run.return_value.pid=123
        self.assertEqual(p.dispatch_replay(sandbox,'/app',1800),123)
        args,kw=sandbox.commands.run.call_args
        self.assertTrue(kw['background'])
        self.assertEqual(kw['timeout'],0)
        self.assertNotIn('nohup',args[0])
        self.assertNotRegex(args[0],r'(?<!&)&(?![&0-9])')
        self.assertIn('timeout -s TERM -k 10s 10800s',args[0])
        self.assertIn('rc.tmp',args[0])
        self.assertIn('mv ',args[0])

    def test_retention_only_explicit_completed_replay(self):
        self.assertTrue(hasattr(p,'retain_for_audit'))
        plan={'actions':[{'call_id':'x'}]};report={'results':[{'call_id':'x'}]}
        self.assertTrue(p.retain_for_audit(True,plan,report,'0'))
        for flag,rep,rc in [(False,report,'0'),(True,report,'1'),(True,{'results':[]},'0')]:
            self.assertFalse(p.retain_for_audit(flag,plan,rep,rc))

if __name__=='__main__':unittest.main()
