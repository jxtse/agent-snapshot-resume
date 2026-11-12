"""Hint persistence through Claude Code auto-compaction (spec persist_hint=true).

Before compaction the hint rides on the fork turn (sentinel present). After compaction the sentinel turn is
summarised away; the proxy must then put the hint, exactly once, on the compact-summary user message of every later
main-agent request, and leave requests without a summary (sub-agents) untouched.
"""
import copy
import json
import unittest

from agent_snapshot_resume.proxy import resume_proxy as proxy

HINT = "Decision: keep the (6,1) mode."
BLK = proxy.hint_block_text(HINT)
SUMMARY = ("This session is being continued from a previous conversation that ran out of context. The summary "
           "below covers the earlier portion of the conversation.\n\nSummary:\n1. Primary Request ...")


def spec(persist=True, mode="hint"):
    return {"mode": mode, "sentinel": "ZZZ_SENT", "hint": HINT if mode == "hint" else None, "persist_hint": persist}


def pre_compaction():
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                                     {"type": "text", "text": proxy.CONTINUE_TXT}]},
        {"role": "assistant", "content": [{"type": "text", "text": proxy.SYNTH_TXT}]},
        {"role": "user", "content": [{"type": "text", "text": "ZZZ_SENT"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "later"}]},
    ]}


def post_compaction(summary_as_string=False, cache=True):
    """Shape of a CC 2.1.280 request after auto-compaction: context reminders, the summary turn, new turns."""
    summ = {"type": "text", "text": SUMMARY}
    if cache:
        summ["cache_control"] = {"type": "ephemeral"}
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nToday's date is 2026-11-10.\n</system-reminder>"},
                                     summ] if not summary_as_string else SUMMARY},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "x"}]},
    ]}


def count(body):
    return proxy.hint_stats(body, HINT)


class PersistHint(unittest.TestCase):
    def test_pre_compaction_hint_once(self):
        out, acts = proxy.rewrite(pre_compaction(), spec())
        self.assertIn("inject_hint", acts)
        self.assertEqual(count(out), (1, 1))

    def test_post_compaction_reinjected_once(self):
        body = post_compaction()
        out, acts = proxy.rewrite(copy.deepcopy(body), spec())
        self.assertIn("sentinel_not_found", acts)
        self.assertTrue(any(a.startswith("reinject_hint_after_compaction@0") for a in acts if isinstance(a, str)))
        self.assertEqual(count(out), (1, 1))
        blocks = out["messages"][0]["content"]
        self.assertEqual(blocks[-1]["text"], BLK)
        # the cache breakpoint stays on the last block of that message, and only once
        self.assertIn("cache_control", blocks[-1])
        self.assertEqual(sum("cache_control" in b for b in blocks), 1)
        # nothing else changed
        self.assertEqual(out["messages"][1:], body["messages"][1:])
        self.assertEqual(blocks[:-1][0], body["messages"][0]["content"][0])
        self.assertEqual(blocks[1]["text"], SUMMARY)

    def test_post_compaction_string_content(self):
        out, acts = proxy.rewrite(post_compaction(summary_as_string=True), spec())
        self.assertEqual(count(out), (1, 1))

    def test_idempotent_no_duplicate(self):
        out, _ = proxy.rewrite(post_compaction(), spec())
        out2, acts2 = proxy.rewrite(copy.deepcopy(out), spec())
        self.assertIn("hint_already_present", acts2)
        self.assertEqual(count(out2), (1, 1))

    def test_summary_quoting_hint_verbatim_not_duplicated(self):
        body = post_compaction()
        body["messages"][0]["content"][1]["text"] += "\nAll user messages: " + HINT
        out, acts = proxy.rewrite(body, spec())
        self.assertIn("hint_already_present", acts)
        self.assertEqual(count(out), (0, 1))

    def test_latest_summary_used_after_second_compaction(self):
        body = post_compaction()
        body["messages"] += [{"role": "assistant", "content": [{"type": "text", "text": "a"}]},
                             {"role": "user", "content": [{"type": "text", "text": SUMMARY + " second"}]}]
        out, acts = proxy.rewrite(body, spec())
        self.assertTrue(any(a == "reinject_hint_after_compaction@4" for a in acts if isinstance(a, str)))
        self.assertEqual(count(out), (1, 1))

    def test_subagent_request_untouched(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "explore the data dir"}]}]}
        out, acts = proxy.rewrite(copy.deepcopy(body), spec())
        self.assertEqual(out, body)
        self.assertIn("no_anchor", acts)

    def test_off_without_persist_or_in_same_mode(self):
        for sp in (spec(persist=False), spec(mode="same")):
            body = post_compaction()
            out, acts = proxy.rewrite(copy.deepcopy(body), sp)
            self.assertEqual(out, body)
            self.assertEqual(acts, ["sentinel_not_found"])

    def test_fingerprint_fields(self):
        out, _ = proxy.rewrite(post_compaction(), spec())
        fp = proxy.request_fingerprint(dict(out, system=[{"type": "text", "text": "sys"}], tools=[{"name": "Bash"}]),
                                       spec())
        self.assertEqual((fp["hint_blocks"], fp["hint_occurrences"], fp["compact_summary"], fp["n_tools"]), (1, 1, True, 1))
        json.dumps(fp)


class BundleFlag(unittest.TestCase):
    def test_persist_needs_hint_mode(self):
        from agent_snapshot_resume import bundle
        with self.assertRaises(SystemExit):
            bundle.build(adapter=None, trial_dir=".", atif_step=1, out=".", mode="same", persist_hint=True)


if __name__ == "__main__":
    unittest.main()
