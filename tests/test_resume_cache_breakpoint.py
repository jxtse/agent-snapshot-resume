"""The trailing prompt-cache breakpoint of a resumed request moves from the dropped resume artifacts to the new
last message (Claude Code 2.1.280 puts it on the post-sentinel environment/date note)."""
import unittest

from agent_snapshot_resume.proxy.resume_proxy import rewrite

CC = {"type": "ephemeral"}


def body(hint=False):
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                                     {"type": "text", "text": "Continue from where you left off."}]},
        {"role": "system", "content": "<total_tokens>1 tokens left</total_tokens>"},
        {"role": "assistant", "content": [{"type": "text", "text": "No response requested."}]},
        {"role": "user", "content": [{"type": "text", "text": "SENTINEL_X"}]},
        {"role": "system", "content": [{"type": "text", "text": "# Environment\nThe date has changed.",
                                        "cache_control": CC}]},
    ]
    spec = {"mode": "hint" if hint else "same", "sentinel": "SENTINEL_X", "hint": "think again" if hint else None}
    return {"messages": msgs}, spec


class CacheBreakpoint(unittest.TestCase):
    def test_moved_to_new_last_message(self):
        b, spec = body()
        new, acts = rewrite(b, spec)
        self.assertIn("move_cache_breakpoint", acts)
        self.assertEqual([m["role"] for m in new["messages"]], ["user", "assistant", "user", "system"])
        self.assertEqual(new["messages"][-1]["content"],
                         [{"type": "text", "text": "<total_tokens>1 tokens left</total_tokens>", "cache_control": CC}])
        n = sum("cache_control" in x for m in new["messages"] for x in (m["content"] if isinstance(m["content"], list) else []))
        self.assertEqual(n, 1)
        self.assertEqual(new["messages"][2]["content"], [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])

    def test_no_breakpoint_no_change(self):
        b, spec = body()
        del b["messages"][-1]["content"][0]["cache_control"]
        new, acts = rewrite(b, spec)
        self.assertNotIn("move_cache_breakpoint", acts)
        self.assertEqual(new["messages"][-1]["content"], "<total_tokens>1 tokens left</total_tokens>")

    def test_hint_mode_still_one_hint(self):
        b, spec = body(hint=True)
        new, acts = rewrite(b, spec)
        self.assertIn("inject_hint", acts)
        self.assertEqual(new["messages"][2]["content"][-1]["text"], "<system-reminder>\nthink again\n</system-reminder>")
