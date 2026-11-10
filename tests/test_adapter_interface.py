"""Adapter contract + Claude Code adapter specifics."""
import unittest

from agent_snapshot_resume.agents import ADAPTERS, get_adapter
from agent_snapshot_resume.agents.base import AgentAdapter
from agent_snapshot_resume.agents.claude_code import CC_SHA256


class AdapterInterface(unittest.TestCase):
    def test_registry(self):
        for name in ADAPTERS:
            a = get_adapter(name)
            self.assertIsInstance(a, AgentAdapter)
            self.assertEqual(a.name, name)
            self.assertIn(a.version, a.binary_sha256)
            self.assertTrue(a.resume_artifacts)
            self.assertTrue(a.harbor_import_path)
        with self.assertRaises(ValueError):
            get_adapter("nope")

    def test_cc_install_pins_version_and_sha256(self):
        a = get_adapter("claude-code")
        self.assertEqual(CC_SHA256["2.1.207"], "85e7e988a392d859f90802ca21fb26e89d3c9ab527f5ed0b08df3955e34d5c83")
        cmds = dict((n, c) for n, c, _ in a.install_commands())
        self.assertIn("claude-code-releases/2.1.207/linux-x64/claude", cmds["download_cc"])
        self.assertIn(CC_SHA256["2.1.207"] + "  /tmp/.asr-claude' | sha256sum -c -", cmds["download_cc"])
        with self.assertRaises(ValueError):
            a.install_commands("9.9.9")

    def test_abstract_methods_enforced(self):
        class Incomplete(AgentAdapter):
            name = "x"
        with self.assertRaises(TypeError):
            Incomplete()
