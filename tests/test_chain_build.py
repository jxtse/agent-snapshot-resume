"""Chained round-N builds: a resumed run's own session seeds the next resume with every hint it carried.

The fixture is built from the synthetic L0 trial: bundle round 1 (one hint), then a *fake* round-1 run dir -
session = prefix + the resume scaffolding Claude Code writes + two new turns, plus the run's own first request
(captured by the resume proxy), proxy spec, bundle copy and run start time. That is exactly the evidence a real
resumed trial leaves behind, so the chained build is exercised end to end without a sandbox.
"""
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agent_snapshot_resume import chain
from agent_snapshot_resume.agents import get_adapter
from agent_snapshot_resume.agents.claude_code import simulate_resume_request
from agent_snapshot_resume.bundle import build
from agent_snapshot_resume.proxy.resume_proxy import hint_block_text, rewrite

ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "examples/fixtures/synthetic-trial"
TASK = ROOT / "examples/fixtures/sum-task"
H1 = "Decision 1: sum the column, not the rows."
H2 = "Decision 2: write the answer to /app/out.txt."


def _uid():
    return str(uuid.uuid4())


def _rec(kind, content, parent=None, **extra):
    rec = {"type": kind, "uuid": _uid(), "timestamp": "2026-01-01T00:00:10.000Z", "parentUuid": parent,
           "sessionId": "00000000-0000-4000-8000-000000000001"}
    if content is not None:
        rec["message"] = {"role": "user" if kind == "user" else "assistant", "content": content}
    rec.update(extra)
    return rec


def make_fake_resumed_run(tmp: Path, *, compact: bool = False) -> dict:
    """A round-1 hinted run: bundle -> simulated first request -> its own session + proxy evidence."""
    adapter = get_adapter("claude-code")
    b1, m1 = build(adapter=adapter, trial_dir=TRIAL, atif_step=4, out=tmp / "b1", task_dir=TASK, mode="hint",
                   hint=H1, persist_hint=True)
    spec1 = json.loads((b1 / "proxy_spec.json").read_text())
    prefix = [json.loads(l) for l in (b1 / "session_trunc.jsonl").read_text().splitlines()]
    raw = simulate_resume_request(prefix, spec1["sentinel"])
    first_request, acts = rewrite(json.loads(json.dumps(raw)), spec1)
    hint_blocks = [b for m in first_request["messages"] for b in
                   (m["content"] if isinstance(m["content"], list) else []) if b.get("text") == hint_block_text(H1)]
    assert len(hint_blocks) == 1, acts

    trial = tmp / "run1" / "sum-task__fake01"
    proj = trial / "agent" / "sessions" / "projects" / "-app"
    proj.mkdir(parents=True)
    recs = list(prefix)
    if compact:
        preserved = [recs[-1]["uuid"]]
        recs.append({"type": "system", "subtype": "compact_boundary", "uuid": _uid(), "timestamp": "2026-01-01T00:05:00.000Z",
                     "compactMetadata": {"trigger": "auto", "preTokens": 140000, "preservedMessages": {"uuids": preserved}}})
        recs.append(_rec("user", "<analysis>What happened so far</analysis><summary>summarised work</summary>",
                         parent=recs[-2]["uuid"], isCompactSummary=True, isVisibleInTranscriptOnly=True))
    last = recs[-1]["uuid"]
    cont = _rec("user", [{"type": "text", "text": chain.CONTINUE_TXT}], parent=last)
    synth = _rec("assistant", [{"type": "text", "text": chain.SYNTH_TXT}], parent=cont["uuid"])
    sent = _rec("user", spec1["sentinel"], parent=synth["uuid"])
    att = _rec("attachment", None, parent=sent["uuid"],
               attachment={"type": "environment", "snapshot": {"workingDirectory": "/app"}})
    a1 = _rec("assistant", [{"type": "thinking", "thinking": "next"}, {"type": "tool_use", "id": "toolu_new1",
                                                                    "name": "Bash", "input": {"command": "ls /app"}}],
              parent=att["uuid"], message_id="msg_new1")
    r1 = _rec("user", [{"type": "tool_result", "tool_use_id": "toolu_new1", "content": "sum.py"}], parent=a1["uuid"])
    a2 = _rec("assistant", [{"type": "thinking", "thinking": "write"}, {"type": "tool_use", "id": "toolu_new2",
                                                                     "name": "Bash", "input": {"command": "echo hi"}}],
              parent=r1["uuid"], message_id="msg_new2")
    r2 = _rec("user", [{"type": "tool_result", "tool_use_id": "toolu_new2", "content": "hi"}], parent=a2["uuid"])
    a3 = _rec("assistant", [{"type": "thinking", "thinking": "done"}, {"type": "tool_use", "id": "toolu_new3",
                                                                    "name": "Bash", "input": {"command": "cat x"}}],
              parent=r2["uuid"], message_id="msg_new3")
    recs += [cont, synth, sent, att, a1, r1, a2, r2, a3]
    sess = proj / f"{spec1.get('session_id') or '00000000-0000-4000-8000-000000000001'}.jsonl"
    sess.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    resume = trial / "agent" / "resume"
    resume.mkdir(parents=True)
    (resume / "proxy_spec.json").write_text(json.dumps(spec1))
    (resume / "bundle.json").write_text(json.dumps(m1))
    (resume / "run_start.json").write_text(json.dumps({"t": m1.get("built_at") and
                                                       float(__import__("datetime").datetime.fromisoformat(
                                                           m1["built_at"]).timestamp()) or 0}))
    rp = trial / "agent" / "resume-proxy" / m1["run_id"]
    rp.mkdir(parents=True)
    (rp / "first_request_rewritten.json").write_text(json.dumps({"body": first_request, "actions": acts}))
    (trial / "result.json").write_text(json.dumps({"task_name": "sum-task", "verifier_result": {"rewards": {"reward": 0.0}},
                                                   "config": {"agent": {"model_name": "test-model"}}}))
    return {"trial": trial, "bundle1": b1, "meta1": m1, "spec1": spec1, "session": sess, "records": recs}


class ChainedBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.fx = make_fake_resumed_run(self.out)

    def _steps(self, trial):
        from agent_snapshot_resume.trace import ir as trace_ir
        adapter = get_adapter("claude-code")
        ir = adapter.load_trace(trial, agent_timeout_s=600.0, task="sum-task", reward=0.0)
        return ir, [s for s in ir["steps"] if s["source"] == "agent"]

    def test_scaffolding_is_stripped_and_chain_reconnected(self):
        recs = self.fx["records"]
        stripped, audit = chain.strip_scaffolding(recs, self.fx["spec1"]["sentinel"])
        self.assertEqual(len(audit), 1)
        self.assertEqual(len(stripped), len(recs) - len(audit[0]["removed_lines"]))
        self.assertEqual(audit[0]["types"][:3], ["user", "assistant", "user"])
        texts = [b.get("text") for r in stripped for b in (r.get("message") or {}).get("content") or []
                 if isinstance(b, dict)]
        self.assertNotIn(chain.CONTINUE_TXT, texts)
        self.assertNotIn(self.fx["spec1"]["sentinel"], texts)
        # the record after the hole points at the record before it, not at a removed uuid
        idx = audit[0]["removed_lines"][0]
        self.assertEqual(stripped[idx]["parentUuid"], audit[0]["reconnected"]["to"])

    def test_hint_materialised_on_the_compaction_summary(self):
        recs = list(self.fx["records"])
        sidx = 500
        recs.append({"type": "system", "subtype": "compact_boundary", "uuid": _uid(),
                     "compactMetadata": {"preservedMessages": {"uuids": []}}})
        summary = _rec("user", "This session is being continued from a previous conversation", isCompactSummary=True)
        recs.append(summary)
        after = _rec("assistant", [{"type": "tool_use", "id": "toolu_after", "name": "Bash",
                                    "input": {"command": "ls"}}], parent=summary["uuid"], message_id="m_x")
        recs.append(after)
        anchors = [{"tool_use_id": "toolu_b1"}]  # injected on the pre-compaction prefix
        out, audit = chain.materialize_hints(recs, [H1], anchors, len(recs))
        self.assertEqual(audit[0]["where"], "compaction_summary")
        self.assertIn(hint_block_text(H1), [b["text"] for b in out[-2]["message"]["content"]])
        self.assertEqual(len(chain.anchors_for(out, [H1], len(out))), 0)  # a summary has no tool result

    def test_round_two_bundle_carries_both_hints(self):
        ir, steps = self._steps(self.fx["trial"])
        cut_step = steps[-1]  # the last agent step of the resumed run
        bdir, meta = build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"], atif_step=cut_step["atif_step_id"],
                           out=self.out / "b2", task_dir=TASK, mode="hint", hint=H2, persist_hint=True)
        spec = json.loads((bdir / "proxy_spec.json").read_text())
        self.assertEqual(spec["hint"], H2)
        self.assertEqual(spec["prior_hints"], [H1])
        self.assertEqual(spec["required_hint_blocks"], [hint_block_text(H1), hint_block_text(H2)])
        self.assertEqual(spec["forbidden_sentinels"], [self.fx["spec1"]["sentinel"]])
        self.assertTrue(spec["persist_hint"])
        self.assertEqual(meta["n_hints"], 2)
        self.assertEqual(meta["chain"]["carried_hints"], 1)
        # the seed carries H1 once (its old injection point) and no scaffolding
        seed = (bdir / "session_trunc.jsonl").read_text()
        seed_texts = [b.get("text") for r in (json.loads(l) for l in seed.splitlines() if l.strip())
                      for b in ((r.get("message") or {}).get("content") or []) if isinstance(b, dict)]
        self.assertEqual(seed_texts.count(hint_block_text(H1)), 1)
        self.assertNotIn(self.fx["spec1"]["sentinel"], seed)
        self.assertNotIn(chain.CONTINUE_TXT, seed)
        # budget comes from the parent run, not from the copied prefix's timestamps
        self.assertEqual(meta["agent_budget_source"], "parent_budget_since_run_start")
        # and the round-2 first request, rewritten, carries both hints exactly once in order
        lines = [json.loads(l) for l in seed.splitlines()]
        body, acts = rewrite(simulate_resume_request(lines, spec["sentinel"]), spec)
        flat = [b.get("text") for m in body["messages"] for b in
                (m["content"] if isinstance(m["content"], list) else [])]
        self.assertEqual(flat.count(hint_block_text(H1)), 1)
        self.assertEqual(flat.count(hint_block_text(H2)), 1)
        self.assertLess(flat.index(hint_block_text(H1)), flat.index(hint_block_text(H2)))
        self.assertNotIn(self.fx["spec1"]["sentinel"], json.dumps(body["messages"]))

    def test_prior_hint_files_must_match_the_run(self):
        ir, steps = self._steps(self.fx["trial"])
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"],
                  atif_step=steps[-1]["atif_step_id"], out=self.out / "bad", task_dir=TASK, mode="hint", hint=H2,
                  prior_hints=["a different hint"])
        prior = [Path(self.fx["bundle1"] / "proxy_spec.json").read_text()]
        prior = [json.loads(prior[0])["hint"]]
        bdir, meta = build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"],
                           atif_step=steps[-1]["atif_step_id"], out=self.out / "ok", task_dir=TASK, mode="hint",
                           hint=H2, prior_hints=prior)
        self.assertEqual(meta["n_hints"], 2)

    def test_non_resumed_trial_refuses_prior_hints(self):
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=TRIAL, atif_step=4, out=self.out / "x", task_dir=TASK,
                  mode="hint", hint=H2, prior_hints=[H1])

    def test_base_snapshot_replays_only_the_tail(self):
        ir, steps = self._steps(self.fx["trial"])
        full, meta = build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"],
                           atif_step=steps[-1]["atif_step_id"], out=self.out / "full", task_dir=TASK, mode="hint",
                           hint=H2)
        plan = json.loads((full / "plan.json").read_text())
        lines = [a["obs_line"] for a in plan["actions"]]
        self.assertTrue(all(l is not None for l in lines))
        self.assertGreater(len(plan["actions"]), 1)
        base = {"snapshot_id": "snap-1", "session_line": lines[0], "ts": "2026-01-01T00:00:00Z",
                "record_path": "runs/x/midrun-snapshots.jsonl", "run_id": meta["run_id"]}
        tail, meta2 = build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"],
                            atif_step=steps[-1]["atif_step_id"], out=self.out / "tail", task_dir=TASK, mode="hint",
                            hint=H2, base_snapshot=base)
        plan2 = json.loads((tail / "plan.json").read_text())
        self.assertEqual(len(plan2["actions"]), len(plan["actions"]) - 1)  # the action observed at that line is in it
        self.assertEqual(meta2["base_snapshot"]["snapshot_id"], "snap-1")
        self.assertEqual(meta2["base_snapshot"]["n_actions_from_snapshot"], 1)
        self.assertEqual(plan2["base_snapshot"], meta2["base_snapshot"])

    def test_base_snapshot_must_precede_the_cut(self):
        ir, steps = self._steps(self.fx["trial"])
        with self.assertRaises(SystemExit):
            build(adapter=get_adapter("claude-code"), trial_dir=self.fx["trial"],
                  atif_step=steps[-1]["atif_step_id"], out=self.out / "y", task_dir=TASK, mode="hint", hint=H2,
                  base_snapshot={"snapshot_id": "s", "session_line": 10 ** 6})


class MultiHintPersistence(unittest.TestCase):
    def spec(self):
        return {"mode": "hint", "sentinel": "ZZZ_RESUME_SENTINEL_test", "hint": H2, "prior_hints": [H1],
                "persist_hint": True,
                "required_hint_blocks": [hint_block_text(H1), hint_block_text(H2)]}

    def body(self, msgs):
        return {"messages": msgs}

    def test_both_hints_are_reinjected_once_on_the_summary(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "start"}]},
                {"role": "user", "content": [{"type": "text", "text": chain.CONTINUE_TXT}]},
                {"role": "user", "content": [{"type": "text", "text": "This session is being continued from a "
                                                                       "previous conversation that ran out of context"}]}]
        out, acts = rewrite(self.body(msgs), self.spec())
        texts = [b["text"] for b in out["messages"][-1]["content"]]
        self.assertEqual(texts.count(hint_block_text(H1)), 1)
        self.assertEqual(texts.count(hint_block_text(H2)), 1)
        self.assertTrue(any(str(a).startswith("reinject_hint_after_compaction") for a in acts))
        # idempotent: a second pass sees both and adds nothing
        out2, acts2 = rewrite(json.loads(json.dumps(out)), self.spec())
        self.assertIn("hint_already_present", acts2)
        self.assertEqual(json.dumps(out2), json.dumps(out))

    def test_missing_prior_hint_fails_closed(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "start"}]},
                {"role": "user", "content": [{"type": "text", "text": "This session is being continued from a "
                                                                       "previous conversation that ran out of context"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
        with self.assertRaises(ValueError):
            # required_hint_blocks cannot be satisfied: H1 was neither seeded nor reinjected
            from agent_snapshot_resume.proxy.resume_proxy import check_chained_history
            check_chained_history({"messages": msgs}, self.spec())
