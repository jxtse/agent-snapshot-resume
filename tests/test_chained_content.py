"""Content parity at a chained cut: the reference is the session, but CC sends some blocks it never persists."""
from agent_snapshot_resume.parity import content_reference
from agent_snapshot_resume.proxy.resume_proxy import check_content

SENTINEL = "ZZZ_RESUME_SENTINEL_t"


def _user(uuid, content, **kw):
    return {"type": "user", "uuid": uuid, "message": {"role": "user", "content": content}, **kw}


def _assistant(uuid, mid, content):
    return {"type": "assistant", "uuid": uuid, "message": {"id": mid, "role": "assistant", "content": content}}


SESSION = [
    _user("u0", "task text"),
    _assistant("a1", "m1", [{"type": "thinking", "thinking": "look", "signature": "sig"},
                            {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]),
    _user("u1", [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]),
    # the agent had no visible output, so Claude Code wrote a meta user record (not a conversation turn) ...
    _user("u2", "[Your previous response had no visible output. Please continue and produce a user-visible response.]",
          isMeta=True),
    # ... and it started the next step's message, whose tool_use is after this cut
    _assistant("a2", "m2", [{"type": "thinking", "thinking": "next", "signature": ""}]),
]


def _request():
    """What Claude Code rebuilds from that prefix: the meta record is sent, the partial message is not."""
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "task text"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "look", "signature": "sig"},
                                          {"type": "tool_use", "id": "c1", "name": "Bash",
                                           "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
                                     {"type": "text", "text": "[Your previous response had no visible output. "
                                                              "Please continue and produce a user-visible response.]"}]},
    ]}


def test_meta_record_and_partial_tail_come_from_the_session_not_the_request():
    ref = content_reference(SESSION)
    notes = []
    problems = check_content(_request(), ref, [], notes)
    assert problems == []
    assert notes == ["tail_thinking_blocks_not_sent:1"]


def test_a_real_block_missing_is_still_a_problem():
    ref = content_reference(SESSION[:3])  # no meta record and no partial tail: nothing to tolerate
    body = _request()
    body["messages"][2]["content"] = body["messages"][2]["content"][:1]  # keep the tool_result, drop the meta text
    assert check_content(body, ref, []) == []
    gone = {"messages": body["messages"][:2]}  # the tool result is gone
    assert check_content(gone, ref, [])


def test_a_foreign_user_text_block_is_still_unexplained():
    body = _request()
    body["messages"][2]["content"] = body["messages"][2]["content"] + [{"type": "text", "text": "unexplained"}]
    problems = check_content(body, content_reference(SESSION), [])
    assert any("unexplained" in p for p in problems), problems
