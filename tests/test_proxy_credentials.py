"""The model key may exist only in the passthrough proxy's memory: never on disk, argv, or a process environment."""
import inspect
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume.agents import base
from agent_snapshot_resume.proxy import runtime


class ProxyCredentials(unittest.TestCase):
    def test_no_disk_key_transition(self):
        text = runtime.MEMORY_PROXY_START
        self.assertNotIn('> "$K"', text)
        self.assertNotIn('read -r KV <', text)
        self.assertNotIn('--key-file /', text)
        src = inspect.getsource(base.SnapshotResumeAgentMixin._start_proxies)
        self.assertIn('passthrough_start_command', src)
        self.assertIn('KEY_ENV_IN_SANDBOX', src)

    def test_key_env_name_is_redacted_by_harbor(self):
        from harbor.utils.env import is_sensitive_env_key
        self.assertTrue(is_sensitive_env_key(runtime.KEY_ENV_IN_SANDBOX))

    def test_upstream_is_shell_quoted(self):
        cmd = runtime.passthrough_start_command("https://api.example.com/v1; rm -rf /")
        self.assertIn("'https://api.example.com/v1; rm -rf /'", cmd)
        with self.assertRaises(ValueError):
            runtime.passthrough_start_command("file:///etc/passwd")

    def test_child_receives_fd_not_secret_file_argv_or_environment(self):
        # Execute the actual generated parent + child shell, with a fake runtime and one loop iteration.
        text = runtime.passthrough_start_command('http://example.invalid', 4013, logdir='LOGDIR_PLACEHOLDER')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'src').mkdir()
            (root / 'rt/bin').mkdir(parents=True)
            rt = root / 'rt/bin/asr-model-proxy'
            rt.write_text('#!/usr/bin/env python3\nimport os,sys\np=sys.argv[sys.argv.index("--key-file")+1]\n'
                          's= open(p).read()\nassert s=="dummy-test-secret"\n'
                          'assert "ASR_PROXY_KEY" not in os.environ\nassert "KV" not in os.environ\n'
                          'assert s not in " ".join(sys.argv)\nprint("fd-only-ok")\n')
            rt.chmod(0o700)
            env = dict(os.environ, ASR_PROXY_KEY='dummy-test-secret')
            parent = (text.split('setsid nohup', 1)[0].replace('/opt/asr-proxy', str(root))
                      .replace('/root/.asr-proxy', str(root / 'state')).replace('LOGDIR_PLACEHOLDER', str(root)))
            generated = subprocess.run(['bash', '-c', parent], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            for p in root.rglob('*'):
                if p.is_file() and p != rt:
                    self.assertNotIn(b'dummy-test-secret', p.read_bytes(), str(p))
            child = ((root / 'state/run-4013.sh').read_text().replace('while true; do', 'for once in 1; do')
                     .replace('  sleep 2', '  :'))
            script = root / 'run.sh'
            script.write_text(child)
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('fd-only-ok', (root / 'proxy-4013.stderr').read_text())
            for p in root.rglob('*'):
                if p.is_file() and p != rt:
                    self.assertNotIn(b'dummy-test-secret', p.read_bytes(), str(p))


class AgentGetsDummyKey(unittest.TestCase):
    def test_resume_env_overrides_any_real_key(self):
        from types import SimpleNamespace
        from pathlib import PurePosixPath
        from agent_snapshot_resume.agents.claude_code import ClaudeCodeSnapshotResume

        a = object.__new__(ClaudeCodeSnapshotResume)
        a._resolve_auth_env = lambda: {"ANTHROPIC_API_KEY": "real-secret", "ANTHROPIC_AUTH_TOKEN": "tok",
                                       "CLAUDE_CODE_OAUTH_TOKEN": "oauth"}
        a._resolved_model_name = lambda: "m"
        a._resolved_env_vars = {"ANTHROPIC_API_KEY": "real-secret-2"}
        a.environment_logs_dir = PurePosixPath("/logs/agent")
        env = ClaudeCodeSnapshotResume._resume_env(a, {"run_id": "r", "model": "m"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "dummy-resume")
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:4031/r/r")
        self.assertNotIn("real-secret", str(env))
