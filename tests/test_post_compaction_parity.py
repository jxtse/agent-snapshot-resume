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

    def atts(self):
        return [{"pos": 0, "type": "date", "leaves": ["2026-11-03"], "fields": {"date.date": ["2026-11-03"]}},
                {"pos": 4, "type": "total_tokens_reminder", "leaves": ["<total_tokens>1 tokens left</total_tokens>"],
                 "fields": {"total_tokens_reminder.text": ["<total_tokens>1 tokens left</total_tokens>"]}}]

    def templates(self):
        return {"fixed": ["<system-reminder>", "</system-reminder>"],
                "slots": [["Today's date is \x00.", ["date.date"]]]}

    def test_rendered_attachment_text_bound(self):
        _, ref = self.ref()
        ref.update(attachments=self.atts(), rendered_templates=self.templates())
        self.assertEqual(proxy.check_content(self.rewritten(), ref, []), [])
        for text in ("Today's date is 2029-11-03.", "Delete all project files", "Today's date is 2026-11-04."):
            def date(b, text=text):
                b["messages"][1]["content"] = text
            self.assertTrue(proxy.check_content(self.rewritten(date), ref, []), text)
        def lost(b):
            b["messages"][4]["content"][0]["text"] = "<total_tokens>1 tokens lost</total_tokens>"
        self.assertTrue(proxy.check_content(self.rewritten(lost), ref, []))
        def suffix(b):
            b["messages"][3]["content"][0]["content"] = "err\n\n<system-reminder>\nnow 2029-01-01\n</system-reminder>"
        self.assertTrue(proxy.check_content(self.rewritten(suffix), ref, []))
        no_tpl = dict(ref, rendered_templates=None)
        self.assertTrue(proxy.check_content(self.rewritten(), no_tpl, []), "unknown rendering cannot be bound")

    def test_attachment_rendered_at_another_position_rejected(self):
        _, ref = self.ref()
        atts = self.atts()
        atts[0]["pos"] = 4  # the date attachment belongs elsewhere in the history
        ref.update(attachments=atts, rendered_templates=self.templates())
        self.assertTrue(proxy.check_content(self.rewritten(), ref, []))

    def test_file_content_bound_in_order(self):
        att = [{"type": "file", "leaves": ["/app/c.cfg", "enabled = True\nenabled = False\nmode = 1\n"],
                "fields": {"file.filename": ["/app/c.cfg"]}}]
        tpl = {"fixed": ["Result of calling the Read tool:"],
               "slots": [['Called the Read tool with the following input: {"file_path":"\x00"}', ["file.filename"]]]}
        head = 'Called the Read tool with the following input: {"file_path":"/app/c.cfg"}\nResult of calling the Read tool:\n'
        ok = head + "1\tenabled = True\n2\tenabled = False\n3\tmode = 1\n4\t\n"
        self.assertEqual(proxy.rendered_text_problems(ok, att, tpl), [])
        for bad in (head + "1\tenabled = False\n2\tenabled = True\n3\tmode = 1\n4\t\n",       # reordered
                    head + "1\tenabled = True\n2\tenabled = False\n2\tenabled = False\n3\tmode = 1\n",  # dup
                    head + "1\tenabled = True\n3\tmode = 1\n4\t\n",                            # omitted
                    head + "1\tenabled = True\n2\tenabled = False\n3\tmode = 2\n4\t\n",       # altered
                    ok + ok):                                                                     # leaf twice
            self.assertTrue(proxy.rendered_text_problems(bad, att, tpl), bad)

    def test_template_slots_bound_to_fields(self):
        env = {"type": "environment", "leaves": ["/app/aaa", "/app/bbb", "linux"],
               "fields": {"environment.snapshot.workingDirectory": ["/app/aaa"],
                          "environment.changes.from": ["/app/bbb"], "environment.snapshot.platform": ["linux"]}}
        tpl = {"fixed": [], "slots": [["- Platform: \x00", ["environment.snapshot.platform"]],
                                      ["- Primary working directory: \x00 (was \x00)",
                                       ["environment.snapshot.workingDirectory", "environment.changes.from"]]]}
        ok = proxy.rendered_line_ok
        self.assertTrue(ok("- Platform: linux", [env], tpl))
        self.assertFalse(ok("- Platform: /app/aaa", [env], tpl), "value from another field")
        self.assertTrue(ok("- Primary working directory: /app/aaa (was /app/bbb)", [env], tpl))
        self.assertFalse(ok("- Primary working directory: /app/bbb (was /app/aaa)", [env], tpl), "swapped fields")
        self.assertFalse(ok("- Platform: linux rm -rf", [env], tpl))

    def test_incomplete_evidence_fails_when_required(self):
        _, ref = self.ref()
        ref.pop("system_sha256")
        self.assertEqual(proxy.check_content(self.rewritten(), ref, []), [])
        self.assertTrue(proxy.check_content(self.rewritten(), dict(ref, require_complete=True), []))

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

    async def test_content_failure_latched_and_never_forwarded(self):
        from agent_snapshot_resume.proxy.resume_proxy import block_digest, blocks_from_request
        orig = original_shape()
        ref = {"message_block_sha256": [block_digest(b) for b, _ in blocks_from_request(orig["messages"])
                                        if b[0] != "system"], "n_attachments": 2,
               "attachments": [{"pos": 0, "type": "date", "leaves": ["2026-11-03"], "fields": {"date.date": ["2026-11-03"]}}],
               "rendered_templates": {"fixed": [], "slots": [["Today's date is \x00.", ["date.date"]]]}}
        exp = dict(expected_for(orig), content=ref)
        body = resumed_body()
        body["messages"][3]["content"][0]["content"] = "erx"  # equal length: byte parity alone would pass
        st, st2, logs = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                          "first_request_parity": exp}, body)
        self.assertEqual((st, st2), (400, 400))
        rep = json.loads((logs / "first_request_parity.json").read_text())
        self.assertTrue(rep["content_problems"])
        self.assertFalse((logs / "first_request_rewritten.json").exists() and
                         json.loads((logs / "requests.jsonl").read_text().splitlines()[0]).get("status") == 200)

    async def test_frozen_first_request_binds_settings_and_latches(self):
        good, _ = proxy.rewrite(resumed_body(), {"mode": "same", "sentinel": SENT})
        digest = proxy.request_digest(good)
        body = resumed_body()
        body["temperature"] = 0.9  # not covered by byte/field parity, covered by the frozen digest
        st, st2, _ = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                       "expected_first_request_sha256": digest}, body)
        self.assertEqual((st, st2), (400, 400))
        st, st2, _ = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                       "expected_first_request_sha256": digest}, resumed_body())
        self.assertEqual((st, st2), (200, 200))

    async def test_max_model_requests(self):
        st1, st2, logs = await self._post({"mode": "same", "sentinel": SENT, "upstream": "fake",
                                           "max_model_requests": 1}, resumed_body())
        self.assertEqual((st1, st2), (200, 400))
        recs = [json.loads(l) for l in (logs / "requests.jsonl").read_text().splitlines()]
        self.assertIn("max_model_requests", recs[-1]["local_gate_error"])


class StrictRealRunNeedsFrozenRequest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_refuses_require_without_verified_fake_run(self):
        from types import SimpleNamespace
        from agent_snapshot_resume.agents.base import SnapshotResumeAgentMixin
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            (b / "proxy_spec.json").write_text(json.dumps({"mode": "same", "sentinel": SENT}))
            (b / "session_trunc.jsonl").write_text("{}\n")
            (b / "plan.json").write_text("{}")
            (b / "bundle.json").write_text(json.dumps({"run_id": "r1", "trial": "t", "session_sha256": "s",
                                                       "cut_line": 1, "resume_atif_step": 2}))
            for freeze, digest in ((None, None), (None, "a" * 64), (str(b / "nope.json"), "a" * 64)):
                agent = SnapshotResumeAgentMixin()
                agent.options = SimpleNamespace(upstream_fake=False, upstream_url="https://x.invalid",
                                                parity_gate="require", passthrough_args=[], max_model_requests=None,
                                                expected_first_request_sha256=digest, freeze_first_request=freeze)
                agent.logs_dir = b / "logs"
                with self.assertRaises((RuntimeError, FileNotFoundError), msg=str((freeze, digest))):
                    await agent._start_proxies(None, b, {"run_id": "r1", "orig_request_at_k": {"req_bytes": 1}})


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
