"""Chained seeding across several compactions: the carried hint must land in the context the model will see."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agent_snapshot_resume import chain  # noqa: E402
from agent_snapshot_resume.equiv import context_lines  # noqa: E402

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


def test_base_snapshot_cleanup_script_is_well_formed():
    """The cleanup runs as a detached python program in the restored clone; extract it and compile it."""
    import re
    src = (Path(__file__).resolve().parent.parent / "src/agent_snapshot_resume/sandbox/source.py").read_text()
    body = re.search(r'kill = """\n(.*?)\n"""', src, re.S).group(1)
    compile(body, "<cleanup>", "exec")
    # it must retry (a restored clone shows the old processes late and the proxy runtime respawns them)
    assert "for _ in range(" in body and "quiet" in body
    assert "/opt/asr-proxy" in body


def test_hint_anchored_on_a_tool_result_gets_its_own_record():
    # Claude Code drops text blocks that ride on a tool-result record when it rebuilds the context, so the hint must
    # be a user turn of its own; no anchor is recorded (there is no tool result in that message).
    records = [
        user("u0", "instruction"),
        {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "a.csv"}]}},
        user("u2", "continue"),
    ]
    seeded, audit, cut = chain.materialize_hints(records, [HINT], [{"tool_use_id": "c1"}], len(records))
    assert audit[0]["where"] == "own_message_after_anchor"
    assert cut == len(records) + 1
    new = seeded[3]
    assert new["type"] == "user" and new["parentUuid"] == "u1"
    assert [b["text"] for b in new["message"]["content"]] == [chain.hint_block(HINT)]
    assert len(chain._occurrences(context_lines(seeded), HINT)) == 1
    assert chain.anchors_for(seeded, [HINT], cut) == []  # the hint is no longer inside the tool-result message


def test_materialise_targets_the_summary_in_context_after_several_compactions():
    records = [
        user("u0", "first instruction"),
        {"type": "assistant", "uuid": "a0", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
        boundary("b1", []),                       # first compaction: summary 1 replaces everything before it
        summary("summary one: the agent compacted early"),
        user("u1", '<tool id="c1">result</tool>', toolUseResult={"id": "c1"}),
        {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]}},
        boundary("b2", []),                       # second compaction: only summary 2 is in the context now
        summary("summary two: the agent compacted again"),
        user("u2", "more work"),
    ]
    ctx = context_lines(records)
    assert [id(r) for r in ctx][0] == id(records[7])          # the later summary is the visible one
    idx = chain._summary_index(records, len(records))
    assert idx == 7                                            # the summary inside the context, not the first one
    anchors = [{"hint": HINT, "tool_use_id": "gone", "msg_index": 0, "block_index": 0}]
    seeded, audit, cut = chain.materialize_hints(records, [HINT], anchors, len(records), persistent=True)
    assert cut == len(records)  # a summary-anchored hint adds no record
    assert audit[0]["where"] == "compaction_summary" and audit[0]["line"] == 7
    # the hint block is now inside the record the model actually sees
    blocks = (seeded[7]["message"]["content"])
    assert isinstance(blocks, str) is False
    assert any(b.get("type") == "text" and HINT in b.get("text", "") for b in blocks)
    assert len(chain._occurrences(context_lines(seeded), HINT)) == 1
