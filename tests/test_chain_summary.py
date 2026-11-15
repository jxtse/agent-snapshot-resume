"""Carried hints across compactions: where the parent's proxy had them, and where the chained run re-places them."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agent_snapshot_resume import chain  # noqa: E402
from agent_snapshot_resume.equiv import context_lines  # noqa: E402
from agent_snapshot_resume.proxy.resume_proxy import hint_block_text  # noqa: E402

HINT = "Decisions:\n1. Use the sparse formulation, not the nullspace basis."
SENTINEL = "<sentinel-xyz>"


def boundary(uuid, keep_uuids):
    return {"type": "system", "subtype": "compact_boundary", "uuid": uuid,
            "compactMetadata": {"preservedMessages": {"uuids": keep_uuids}}}


def summary(text):
    return {"type": "user", "uuid": "s-" + text[:6], "isCompactSummary": True,
            "message": {"role": "user", "content": text}}


def user(uuid, text, **kw):
    return {"type": "user", "uuid": uuid, "message": {"role": "user", "content": text}, **kw}


def pre_compaction_records():
    return [
        user("u0", "instruction"),
        {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        user("u1", [{"type": "tool_result", "tool_use_id": "c1", "content": "a.csv"}]),
    ]


def parent_evidence(persist_hint=True, anchor="c1"):
    """The parent's own first request: its proxy appended the hint block to the message that carried the anchor."""
    body = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "instruction"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": anchor, "content": "a.csv"},
                                     {"type": "text", "text": hint_block_text(HINT)}]}]}
    return {"first_request": {"body": body}, "spec": {"persist_hint": persist_hint, "sentinel": SENTINEL,
                                                     "mode": "hint", "hint": HINT}}


def test_base_snapshot_cleanup_script_is_well_formed():
    """The cleanup runs as a detached python program in the restored clone; extract it and compile it."""
    import re
    src = (Path(__file__).resolve().parent.parent / "src/agent_snapshot_resume/sandbox/source.py").read_text()
    body = re.search(r'kill = """\n(.*?)\n"""', src, re.S).group(1)
    compile(body, "<cleanup>", "exec")
    # it must retry (a restored clone shows the old processes late and the proxy runtime respawns them)
    assert "for _ in range(" in body and "quiet" in body
    assert "/opt/asr-proxy" in body


def test_pre_compaction_hint_keeps_its_anchor_tool_result():
    recs = pre_compaction_records()
    pos = chain.carried_hint_positions(parent_evidence(), recs, len(recs), [HINT])
    assert len(pos) == 1
    assert pos[0]["tool_use_id"] == "c1" and not pos[0].get("on_summary")
    assert pos[0]["block"] == hint_block_text(HINT)


def test_after_several_compactions_the_hint_goes_on_the_summary_in_context():
    records = [
        user("u0", "first instruction"),
        {"type": "assistant", "uuid": "a0", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "ok"}]}},
        boundary("b1", []),                       # first compaction: summary 1 replaces everything before it
        summary("summary one: the agent compacted early"),
        user("u1", '<tool id="c1">result</tool>', toolUseResult={"id": "c1"}),
        {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        boundary("b2", []),                       # second compaction: only summary 2 is in the context now
        summary("summary two: the agent compacted again"),
        user("u2", "more work"),
    ]
    assert [id(r) for r in context_lines(records)][0] == id(records[7])  # the later summary is the visible one
    assert chain._summary_index(records, len(records)) == 7              # in the context, not the first one
    pos = chain.carried_hint_positions(parent_evidence(persist_hint=True), records, len(records), [HINT])
    assert pos[0]["on_summary"] is True
    # the tool result the parent injected on is still recorded, for the audit trail only
    assert pos[0]["tool_use_id"] == "c1"


def test_compacted_away_hint_without_persistence_is_refused():
    """The parent's own proxy dropped the hint at its compactions: no chained run can invent it back."""
    records = [
        user("u0", "instruction"),
        boundary("b1", []),
        summary("summary one: everything before it is gone"),
        user("u1", "after"),
    ]
    with pytest.raises(ValueError, match="did not persist hints"):
        chain.carried_hint_positions(parent_evidence(persist_hint=False), records, len(records), [HINT])


PREFIX = "This session is being continued from a previous conversation"


def summary_records():
    return [
        user("u0", "instruction"),
        boundary("b0", []),
        summary(f"{PREFIX}. summary zero: the agent compacted before its resume"),
        {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        user("u1", [{"type": "tool_result", "tool_use_id": "c1", "content": "a.csv"}]),
        user("u2", "more work"),
    ]


def test_the_same_summary_keeps_the_hint_on_its_anchor():
    """The run resumed from a compaction, injected the hint on a tool result, and did not compact again."""
    recs = summary_records()
    ev = parent_evidence(persist_hint=True)
    ev["first_request"]["body"]["messages"].insert(
        0, {"role": "user", "content": f"{PREFIX}. summary zero: the agent compacted before its resume"})
    pos = chain.carried_hint_positions(ev, recs, len(recs), [HINT])
    assert not pos[0].get("on_summary")
    assert pos[0]["tool_use_id"] == "c1"


def test_a_later_compaction_moves_the_hint_to_the_new_summary():
    recs = summary_records()
    recs += [boundary("b1", []), summary(f"{PREFIX}. summary one: the agent compacted again"),
             user("u3", "after")]
    ev = parent_evidence(persist_hint=True)
    ev["first_request"]["body"]["messages"].insert(
        0, {"role": "user", "content": f"{PREFIX}. summary zero: the agent compacted before its resume"})
    pos = chain.carried_hint_positions(ev, recs, len(recs), [HINT])
    assert pos[0]["on_summary"] is True
