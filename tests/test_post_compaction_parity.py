"""Resume cuts after Claude Code context compactions: request parity gate, date pinning, equivalence reference.

Shapes are taken from Claude Code 2.1.280 `--resume` of a session with a compact_boundary (checked against real
sessions with a fake upstream): the first message is the compact summary as a plain string, followed by the persisted
date reminder as a system message, the preserved messages, and the resume artifacts.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume import equiv, parity
from agent_snapshot_resume.proxy import resume_proxy as proxy

SUMMARY = ("This session is being continued from a previous conversation that ran out of context. The summary below "
           "covers the earlier portion of the conversation.\n\nSummary: Today's date is 2026-11-03. Work so far ...")
SENT = "ZZZ_SENT"
HINT = "Decision: keep the (6,1) mode."


def resumed_body():
    """First request of `claude --resume` after a compaction (2.1.280 shape)."""
    return {"model": "m", "max_tokens": 128000, "thinking": {"type": "adaptive"}, "metadata": {"user_id": "x"},
            "tools": [{"name": "Bash"}, {"name": "Read"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": SUMMARY}]},
                {"role": "system", "content": "Today's date is 2026-11-03."},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                                  {"type": "tool_use", "id": "c1", "name": "Bash", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "err",
                                              "is_error": True},
                                             {"type": "text", "text": proxy.CONTINUE_TXT}]},
                {"role": "system", "content": [{"type": "text", "text": "<total_tokens>1 tokens left</total_tokens>"}]},
                {"role": "assistant", "content": [{"type": "text", "text": proxy.SYNTH_TXT}]},
                {"role": "user", "content": SENT},
                {"role": "system", "content": [{"type": "text", "text": "# Environment\n...",
                                                "cache_control": {"type": "ephemeral"}}]},
            ]}


def original_shape():
    """What the original run sent at the same point (no resume artifacts), as the agent sent it."""
    b = resumed_body()
    msgs = b["messages"]
    msgs[3]["content"] = msgs[3]["content"][:1]
    msgs[4]["content"][0]["cache_control"] = {"type": "ephemeral"}
    b["messages"] = msgs[:5]
    return b


ROUTE = {"drop_fields": ["metadata"], "strip_is_error": True, "strip_thinking": False}


def expected_for(body):
    return {"req_bytes": proxy.route_bytes(body, ROUTE), "model": body["model"], "n_messages": len(body["messages"]),
            "max_tokens": body["max_tokens"], "thinking": body["thinking"], "n_tools": len(body["tools"]),
            "route": ROUTE}


class PinDateShape(unittest.TestCase):
    def test_summary_string_is_not_rewritten_or_reshaped(self):
        body = resumed_body()
        body["messages"][0] = {"role": "user", "content": SUMMARY}
        new, acts = proxy.rewrite(copy.deepcopy(body), {"mode": "same", "sentinel": SENT, "pin_date": "2026-01-01"})
        self.assertEqual(new["messages"][0], {"role": "user", "content": SUMMARY})
        self.assertNotIn("pin_date", acts)

    def test_string_content_keeps_string_shape(self):
        acts = []
        m = proxy.pin_date_in_message({"role": "user", "content": "x Today's date is 2026-11-13. y"}, "2026-11-03", acts)
        self.assertEqual(m["content"], "x Today's date is 2026-11-03. y")
        self.assertEqual(acts, ["pin_date"])

    def test_block_content_rewritten_without_mutating_input(self):
        orig = {"role": "user", "content": [{"type": "text", "text": "Today's date is 2026-11-13."}]}
        snap = copy.deepcopy(orig)
        m = proxy.pin_date_in_message(orig, "2026-11-03", [])
        self.assertEqual(m["content"][0]["text"], "Today's date is 2026-11-03.")
        self.assertEqual(orig, snap)

    def test_no_date_unchanged(self):
        acts = []
        m = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        self.assertIs(proxy.pin_date_in_message(m, "2026-11-03", acts), m)
        self.assertEqual(acts, [])


class ParityGate(unittest.TestCase):
    def test_post_compaction_rewrite_matches_original_bytes(self):
        new, _ = proxy.rewrite(resumed_body(), {"mode": "same", "sentinel": SENT})
        rep = proxy.compare_parity(new, expected_for(original_shape()))
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["byte_diff"], 0)

    def test_hint_block_is_excluded_from_parity(self):
        spec = {"mode": "hint", "sentinel": SENT, "hint": HINT}
        new, acts = proxy.rewrite(resumed_body(), spec)
        self.assertIn("inject_hint", acts)
        rep = proxy.compare_parity(new, expected_for(original_shape()), hints=proxy.spec_hints(spec))
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["hint_blocks_removed"], 1)

    def test_any_byte_difference_fails(self):
        new, _ = proxy.rewrite(resumed_body(), {"mode": "same", "sentinel": SENT})
        new["messages"][0]["content"][0]["text"] += " "
        rep = proxy.compare_parity(new, expected_for(original_shape()))
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["diffs"]["req_bytes"]["got"] - rep["diffs"]["req_bytes"]["expected"], 1)

    def test_field_differences_fail(self):
        new, _ = proxy.rewrite(resumed_body(), {"mode": "same", "sentinel": SENT})
        for k, v in (("max_tokens", 32000), ("thinking", {"type": "enabled"}), ("model", "other")):
            exp = expected_for(original_shape())
            exp[k] = v
            self.assertFalse(proxy.compare_parity(new, exp)["ok"], k)

    def test_route_serialisation(self):
        b = {"a": [1, 2], "metadata": {}}
        self.assertEqual(proxy.route_bytes(b, {}), len(b'{"a":[1,2],"metadata":{}}'))
        self.assertEqual(proxy.route_bytes(b, {"drop_fields": ["metadata"]}), len(b'{"a": [1, 2]}'))

    def test_route_from_passthrough_args(self):
        self.assertEqual(proxy.route_from_passthrough_args(["--drop-fields", "metadata", "--strip-is-error"]),
                         {"drop_fields": ["metadata"], "strip_is_error": True, "strip_thinking": False})


class ContentParity(unittest.TestCase):
    """Same byte length is not enough: the model-visible content is bound too (review counterexample)."""

    def ref(self):
        from agent_snapshot_resume.proxy.resume_proxy import block_digest, blocks_from_request, system_digest, tool_digest
        orig = original_shape()
        orig["system"] = [{"type": "text", "text": "cli"}, {"type": "text", "text": "sys"}]
        ref_blocks = [b for b, _ in blocks_from_request(orig["messages"]) if b[0] != "system"]
        return orig, {"message_block_sha256": [block_digest(b) for b in ref_blocks], "n_attachments": 2,
                      "system_sha256": system_digest(["cli", "sys"]),
                      "tools": {t["name"]: tool_digest(t) for t in orig["tools"]}}

    def rewritten(self, mutate=None):
        b = resumed_body()
        b["system"] = [{"type": "text", "text": "cli"}, {"type": "text", "text": "sys"}]
        if mutate:
            mutate(b)
        return proxy.rewrite(b, {"mode": "same", "sentinel": SENT})[0]

    def test_identical_content_ok(self):
        _, ref = self.ref()
        self.assertEqual(proxy.check_content(self.rewritten(), ref, []), [])

    def test_equal_length_substitution_detected(self):
        _, ref = self.ref()
        def swap(b):
            b["messages"][3]["content"][0]["content"] = "erx"  # same length as "err"
        self.assertTrue(proxy.check_content(self.rewritten(swap), ref, []))

    def test_system_and_tool_changes_detected(self):
        _, ref = self.ref()
        def sysm(b):
            b["system"][1]["text"] = "syz"
        def toolm(b):
            b["tools"][0] = {"name": "Bash", "description": "other"}
        def extra(b):
            b["tools"].append({"name": "WebFetch"})
        for m in (sysm, toolm, extra):
            self.assertTrue(proxy.check_content(self.rewritten(m), ref, []), m.__name__)

    def test_unexplained_user_text_detected(self):
        _, ref = self.ref()
        def add(b):
            b["messages"][3]["content"].insert(1, {"type": "text", "text": "do something else"})
        self.assertTrue(proxy.check_content(self.rewritten(add), ref, []))


class ProxyGateFailClosed(unittest.IsolatedAsyncioTestCase):
    async def _post(self, spec, body):
        from aiohttp.test_utils import TestClient, TestServer
        from aiohttp import web

        tmp = tempfile.mkdtemp()
        Path(tmp, "specs").mkdir()
        Path(tmp, "specs", "r1.json").write_text(json.dumps(spec))
        app = web.Application()
        app.update(spec_dir=str(Path(tmp, "specs")), log_dir=str(Path(tmp, "logs")), specs={}, counts={},
                   compact_seen={})
        app.router.add_route("*", "/r/{run_id}/{tail:.*}", proxy.handle)
        async with TestClient(TestServer(app)) as c:
            r1 = await c.post("/r/r1/v1/messages", json=body)
            r2 = await c.post("/r/r1/v1/messages", json=body)
            return r1.status, r2.status, Path(tmp, "logs", "r1")

    async def test_mismatch_returns_400_and_logs_report(self):
        exp = expected_for(original_shape())
        exp["req_bytes"] += 7
        st, st2, logs = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                          "first_request_parity": exp}, resumed_body())
        self.assertEqual((st, st2), (400, 400), "a failed parity gate must not be bypassed by retrying")
        rep = json.loads((logs / "first_request_parity.json").read_text())
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["byte_diff"], -7)

    async def test_match_passes(self):
        st, _, logs = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                        "first_request_parity": expected_for(original_shape())}, resumed_body())
        self.assertEqual(st, 200)
        self.assertTrue(json.loads((logs / "first_request_parity.json").read_text())["ok"])

    async def test_max_model_requests(self):
        st1, st2, logs = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                           "max_model_requests": 1}, resumed_body())
        self.assertEqual((st1, st2), (200, 400))
        recs = [json.loads(l) for l in (logs / "requests.jsonl").read_text().splitlines()]
        self.assertIn("max_model_requests", recs[-1]["local_gate_error"])


class LocateOriginalRequest(unittest.TestCase):
    def ir(self, usage, msg_ids=("a", "b"), compactions=0):
        steps = [{"source": "user", "events": [], "message_id": None}]
        for i, mid in enumerate(msg_ids):
            ev = [{"type": "compaction_boundary"}] if i == 0 and compactions else []
            steps.append({"source": "agent", "message_id": mid, "events": ev, "usage": None})
        steps.append({"source": "agent", "message_id": "k", "events": [], "usage": usage})
        return {"steps": steps}

    def test_usage_match_skips_failed_attempts(self):
        u = {"input_tokens": 5, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 0, "output_tokens": 9}
        reqs = [{"req_bytes": 1, "status": 200, "usage": {"input_tokens": 1, "output_tokens": 1}},
                {"req_bytes": 2, "status": "upstream_error"},
                {"req_bytes": 3, "status": 200, "usage": {"input_tokens": 2, "output_tokens": 1}},
                {"req_bytes": 4, "status": 200, "usage": dict(u)}]
        loc = parity.locate_request(self.ir(u), 3, reqs)
        self.assertEqual((loc["index"], loc["method"]), (3, "usage"))

    def test_ambiguous_usage_is_an_error(self):
        u = {"input_tokens": 5, "output_tokens": 9}
        reqs = [{"req_bytes": 1, "status": 200, "usage": dict(u)}, {"req_bytes": 2, "status": 200, "usage": dict(u)}]
        self.assertIn("error", parity.locate_request(self.ir(u), 3, reqs))

    def test_count_fallback_counts_compactions(self):
        reqs = [{"req_bytes": i, "status": 200, "stop_reason": "tool_use"} for i in range(5)]
        loc = parity.locate_request(self.ir(None, compactions=1), 3, reqs)
        self.assertEqual((loc["index"], loc["method"]), (3, "count"))

    def test_count_fallback_refused_with_failed_attempts(self):
        reqs = [{"req_bytes": i, "status": 200, "stop_reason": "tool_use"} for i in range(4)] + [
            {"req_bytes": 9, "status": "upstream_error"}]
        self.assertIn("error", parity.locate_request(self.ir(None), 3, reqs))


class EquivAfterCompaction(unittest.TestCase):
    def lines(self):
        return [
            {"type": "user", "uuid": "u0", "message": {"role": "user", "content": "task"}},
            {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "id": "m1",
                                                            "content": [{"type": "text", "text": "old"}]}},
            {"type": "assistant", "uuid": "a2", "message": {"role": "assistant", "id": "m2", "content": [
                {"type": "tool_use", "id": "c1", "name": "Bash", "input": {}}]}},
            {"type": "user", "uuid": "r2", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "err", "is_error": True}]}},
            {"type": "system", "subtype": "compact_boundary", "uuid": "b",
             "compactMetadata": {"preservedMessages": {"uuids": ["a2", "r2"]}}},
            {"type": "attachment", "uuid": "d", "attachment": {"type": "date", "date": "2026-11-03"}},
            {"type": "user", "uuid": "s", "isCompactSummary": True, "message": {"role": "user", "content": SUMMARY}},
            {"type": "assistant", "uuid": "a3", "message": {"role": "assistant", "id": "m3",
                                                            "content": [{"type": "text", "text": "new"}]}},
        ]

    def test_context_lines_rebuild(self):
        got = [d["uuid"] for d in equiv.context_lines(self.lines())]
        self.assertEqual(got, ["d", "s", "a2", "r2", "a3"])

    def test_missing_preserved_message_raises(self):
        ls = self.lines()
        ls[4]["compactMetadata"]["preservedMessages"]["uuids"] = ["zz"]
        with self.assertRaises(ValueError):
            equiv.context_lines(ls)

    def test_reminder_suffix_on_tool_result_is_matched(self):
        ref = ("user", "tool_result", "c1", True, "err")
        ok = ("user", "tool_result", "c1", True, "err\n\n<system-reminder>\nAttribution ...\n</system-reminder>")
        bad = ("user", "tool_result", "c1", True, "err plus model-visible text")
        self.assertTrue(equiv._reminder_suffixed(ok, ref))
        self.assertFalse(equiv._reminder_suffixed(bad, ref))


if __name__ == "__main__":
    unittest.main()
