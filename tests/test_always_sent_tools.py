"""Web tools are sent on every Claude Code request but never snapshotted; that must not read as an extra tool."""
from agent_snapshot_resume.proxy.resume_proxy import ALWAYS_SENT_TOOLS, check_content, tool_digest

A = {"name": "Bash", "description": "run", "input_schema": {"type": "object"}}
B = {"name": "Read", "description": "read", "input_schema": {"type": "object"}}
REF = {"tools": {t["name"]: tool_digest(t) for t in (A, B)}}
WEB = [{"name": n, "description": "web", "input_schema": {"type": "object"}} for n in sorted(ALWAYS_SENT_TOOLS)]


def body(tools):
    return {"tools": tools, "system": [], "messages": []}


def test_web_tools_absent_from_the_snapshot_are_allowed():
    assert check_content(body([A, B, *WEB]), REF, []) == []


def test_other_unexpected_tools_are_still_reported():
    problems = check_content(body([A, B, {"name": "Task", "description": "x", "input_schema": {}}]), REF, [])
    assert problems == ["unexpected tools ['Task']"]


def test_a_snapshotted_tool_that_changed_is_still_reported():
    changed = {**A, "description": "different"}
    problems = check_content(body([changed, B, *WEB]), REF, [])
    assert problems == ["tool Bash missing or different"]
