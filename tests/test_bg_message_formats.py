import unittest

from agent_snapshot_resume.trace.claude_code import _BG_RE


class BackgroundMessage(unittest.TestCase):
    def test_explicit(self):
        m = _BG_RE.search("Command running in background with ID: b1. Output is being written to: /tmp/t/b1.output. x")
        self.assertEqual((m.group(1) or m.group(2), m.group(3)), ("b1", "/tmp/t/b1.output"))

    def test_auto_backgrounded_cc280(self):
        txt = ("Command did not complete within its 120s timeout and was moved to the background (ID: bua1qmh6q). "
               "Output is being written to: /tmp/claude-0/-app/s/tasks/bua1qmh6q.output. You will be notified when it completes.")
        m = _BG_RE.search(txt)
        self.assertEqual((m.group(1) or m.group(2), m.group(3)), ("bua1qmh6q", "/tmp/claude-0/-app/s/tasks/bua1qmh6q.output"))


if __name__ == "__main__":
    unittest.main()
