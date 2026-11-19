"""Offline native sessions and request gates. No external CLI, credentials or sockets."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_snapshot_resume import bundle, equiv, parity
from agent_snapshot_resume.agents.codex import CodexAdapter, codex_history, decode_output, tool_action
from agent_snapshot_resume.agents.pi import PiAdapter, active_branch, tool_action as pi_tool
from agent_snapshot_resume.proxy import native_wire as wire
from agent_snapshot_resume.proxy import resume_proxy as proxy
from agent_snapshot_resume.replay.replay_inside import edit_original_spans


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\r\n" for row in rows)
    path.write_bytes(data)
    return data


def codex_trial(path, monkeypatch):
    from agent_snapshot_resume.agents import codex
    catalog = json.dumps({"models": [{"slug": "some-model", "truncation_policy": {"mode": "tokens", "limit": 10000}}]}).encode()
    monkeypatch.setattr(codex, "CATALOG_SHA256", hashlib.sha256(catalog).hexdigest())
    (path / "agent").mkdir()
    (path / "agent/model-catalog.json").write_bytes(catalog)

    def row(typ, **payload):
        return dict(timestamp="2026-01-01T00:00:01Z", type=typ, payload=payload)
    usage = dict(input_tokens=71, output_tokens=7)
    rows = [
        row("session_meta", id="thread", cwd="/work", model_provider="fixture", cli_version="0.154.0",
            base_instructions={"text": "base"}),
        row("turn_context", cwd="/work", model="some-model", effort="low"),
        row("response_item", type="message", role="user", content=[{"text": "task"}]),
        row("response_item", type="function_call", call_id="c", name="exec_command", arguments='{"cmd":"run"}'),
        row("response_item", type="function_call_output", call_id="c",
            output="Chunk ID: 1\nWall time: 0.1 seconds\nProcess running with session ID 99\nOutput:\nstart"),
        row("event_msg", type="token_count"),
        row("response_item", type="function_call", name="write_stdin", call_id="poll",
            arguments='{"session_id":99,"chars":""}'),
        row("response_item", type="function_call_output", call_id="poll",
            output="Chunk ID: 2\nProcess exited with code 0\nOutput:\ndone"),
        row("event_msg", type="token_count"),
        row("compacted", replacement_history=[{"role": "user", "type": "message", "content": "summary"}]),
        row("event_msg", type="item_completed", item={"type": "Reasoning"}),
        row("response_item", type="reasoning", content=[]),
        row("response_item", type="message", role="assistant", content=[{"text": "done"}]),
        row("token_usage_record", usage=usage, response_id="r"),
    ]
    original = write_lines(path/"agent/sessions/2026/01/01/rollout-demo.jsonl", rows)
    write_lines(path/"agent/model-proxy/requests-1.jsonl",
                [{"status": 200, "usage": usage, "model": "some-model", "req_bytes": 300}])
    return rows, original


def test_codex_cut_compaction_and_processes(tmp_path, monkeypatch):
    rows, data = codex_trial(tmp_path, monkeypatch)
    adapter = CodexAdapter()
    ir = adapter.load_trace(tmp_path, 60)
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=4, agent_timeout=60, out=tmp_path/"out")
    assert meta["cut_line"] == 10 and meta["n_compactions_before_cut"] == 1
    assert (b/"session_trunc.jsonl").read_bytes() == b"".join(data.splitlines(True)[:10])
    plan = json.loads((b/"plan.json").read_text())
    first, = plan["actions"]
    assert first["tool"] == "Bash" and first["bg_finished_before_k"]
    assert first["orig_text"] == "startdone" and first["side_effect"] == "exec"
    content = meta["orig_request_at_k"]["content"]
    assert content["items"] == [wire.digest(rows[9]["payload"]["replacement_history"][0])]
    projected = {"model": "some-model", "instructions": "base", "input": codex_history(rows[:10], 10000)}
    partial = wire.compare_native(projected, content)
    assert partial["verdict"] == "incomplete_evidence" and not partial["ok"] and not partial["content_problems"]
    different = wire.compare_native({**projected, "model": "other"}, content)
    assert different["verdict"] == "mismatch" and not different["evidence_complete"]
    dest = b/"capture.json"
    dest.write_text(json.dumps(projected))
    assert not equiv.check(b/"session_trunc.jsonl", meta["cut_line"], dest)["equivalent"]
    with pytest.raises(ValueError, match="replacement history"):
        codex_history([{"type": "compacted", "payload": {}}], 10000)
    assert adapter.native_reference(tmp_path, ir, 1, [])[0] is None
    log = tmp_path/"agent/model-proxy/requests-1.jsonl"
    log.write_bytes(log.read_bytes()*2)
    assert adapter.native_reference(tmp_path, ir, 3, [])[0] is None
    full = wire.body_reference("responses", projected)
    write_lines(tmp_path/"agent/model-proxy/native-reference.jsonl",
                [dict(entry_id="r", session_id="thread", content=full)])
    assert adapter.native_reference(tmp_path, ir, 3, [])[0]["content"] == full


def test_codex_mappings():
    shell = {"type": "function_call", "name": "shell", "arguments": '{"command":["ls","a b"],"workdir":"sub"}'}
    assert tool_action(shell, "/base") == ("Bash", {"command": "ls 'a b'"}, "read_only", "/base/sub")
    patch = {"type": "custom_tool_call", "name": "apply_patch", "input": "line one\nline two"}
    assert tool_action(patch, "/base")[1]["command"].startswith("codex --codex-run-as-apply-patch")
    for unknown in (dict(patch, name="extension"), dict(shell, name="mystery")):
        assert tool_action(unknown, "/base")[2] == "other"
    assert decode_output("Chunk ID: abc\nProcess exited with code 7\nFinal output:\nmissing")[1] == 7
    assert decode_output("without header")[0] == "without header"


def pi_rows():
    header = dict(type="session", id="session", version=3, cwd="/task")
    assistant = dict(role="assistant", model="fixture", provider="local", api="openai-completions")
    rows = [header]
    def add(entry, parent, typ, **rest):
        rows.append(dict(type=typ, id=entry, parentId=parent, **rest))
    add("u", None, "message", message=dict(role="user", content="instruction"))
    add("w", "u", "message", message=dict(assistant, content=[
        dict(type="toolCall", name="write", id="call", arguments={"path": "out", "content": "ok"})]))
    add("o", "w", "message", message=dict(role="toolResult", toolCallId="call", isError=False,
                                          durationMs=99, content=[dict(type="text", text="written")]))
    add("old", "o", "message", message=dict(assistant, content=[dict(type="text", text="old branch")]))
    add("a", "o", "message", message=dict(assistant, content=[dict(type="text", text="active branch")]))
    for i, r in enumerate(rows):
        r["timestamp"] = f"2026-01-01T00:00:0{i}Z"
    return rows


def test_pi_tree_reference_and_equiv(tmp_path):
    rows = pi_rows()
    session = tmp_path / "agent/pi/sessions/session.jsonl"
    data = write_lines(session, rows)
    (tmp_path / "agent/pi/version.txt").write_text("1.4.0")
    adapter = PiAdapter()
    ir = adapter.load_trace(tmp_path, 60)
    assert [s["message_id"] for s in ir["steps"]] == ["u", "w", "a"]
    original = {"messages": [{"role": "user", "content": "instruction"}], "model": "fixture", "tools": []}
    ref = wire.body_reference("chat", original)
    write_lines(tmp_path/"agent/model-proxy/native-reference.jsonl",
                [dict(entry_id="a", session_id="session", content=ref)])
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=3, agent_timeout=60, out=tmp_path/"out")
    assert (b/"session_trunc.jsonl").read_bytes() == b"".join(data.splitlines(True)[:4])
    dest = b/"capture.json"
    dest.write_text(json.dumps(original))
    assert parity.check_files(b, dest)["ok"]
    assert equiv.check(b/"session_trunc.jsonl", meta["cut_line"], dest)["equivalent"]
    (b/"session_trunc.jsonl").write_bytes(data)
    assert not equiv.check(b/"session_trunc.jsonl", meta["cut_line"], dest)["equivalent"]
    rows[4]["message"] = rows[2]["message"]
    write_lines(session, rows)
    ir = adapter.load_trace(tmp_path, 60)
    with pytest.raises(ValueError, match="workspace side effects"):
        adapter.cut_line([json.dumps(r) for r in rows], ir, 2)
    rows[-1]["parentId"] = "missing"
    with pytest.raises(ValueError, match="invalid Pi parent"):
        list(active_branch(rows))


def test_pi_tools(tmp_path):
    assert pi_tool("read", {"path": "abc"}, "/")[1:] == ({"file_path": "/abc"}, "read_only")
    assert pi_tool("bash", {"command": "ls", "timeout": 2}, "/")[1]["timeout"] == 2000
    assert pi_tool("edit", {"path": "abc", "oldText": "x", "newText": "y"}, "/")[0] == "Edit"
    name, args, _ = pi_tool("edit", {"path": "abc", "edits": [
        {"oldText": "first", "newText": "second"}, {"oldText": "second", "newText": "third"}]}, str(tmp_path))
    assert name == "MultiEdit"
    path = Path(args["file_path"])
    path.write_text("first second")
    assert edit_original_spans(path, args["edits"], False) == ("applied", "exact")
    assert path.read_text() == "second third"
    assert edit_original_spans(path, args["edits"], True)[1] == "error_reproduced"
    assert pi_tool("extension", {}, "/")[2] == "other"


@pytest.mark.parametrize("format", ["chat", "responses"])
def test_wire_rewrite_and_full_gate(format):
    field = wire.FIELDS[format]
    item = wire.user_item(format, "hello")
    original = {field: [item], "model": "m", "tools": [{"name": "x"}], "metadata": {"id": "x"}}
    sentinel = wire.user_item(format, "UNIQUE")
    spec = {"wire": format, "sentinel": "UNIQUE", "mode": "hint", "hint": "hint"}
    ref = wire.body_reference(format, original)
    body, _ = proxy.rewrite({**original, field: [item, sentinel]}, spec, first=True)
    assert wire.compare_native(body, ref, ["hint"])["ok"]
    assert not wire.compare_native({**body, "metadata": {"id": "y"}}, ref, ["hint"])["ok"]
    assert not wire.compare_native({**body, field: list(reversed(body[field]))}, ref, ["hint"])["ok"]
    assert not wire.compare_native(body, dict(ref, complete=False), ["hint"])["ok"]
    for history in ([item], [sentinel, item, sentinel]):
        with pytest.raises(ValueError, match="exactly one"):
            proxy.rewrite({field: history}, spec, first=True)
    later = [item, sentinel, wire.user_item(format, "later")]
    assert proxy.rewrite({field: later}, spec)[0][field][-1] == later[-1]
    assert proxy.rewrite(original, spec)[0] == original
    assert proxy.native_fake_reply(format, {**original, "stream": True}).content_type == "text/event-stream"
    assert proxy.native_fake_reply(format, original).content_type == "application/json"


@pytest.mark.parametrize("format", ["chat", "responses"])
def test_proxy_routes_and_latches(tmp_path, format):
    field = wire.FIELDS[format]
    orig = {field: [wire.user_item(format, "some task")], "model": "fixture"}
    spec = {"upstream": "fake", "wire": format, "mode": "same", "sentinel": "S",
            "first_request_parity": {"content": wire.body_reference(format, orig)}}
    app = dict(spec_dir=tmp_path, log_dir=tmp_path/"logs", counts={}, specs={})
    def invoke(route, body):
        req = SimpleNamespace(app=app, method="POST", headers={}, match_info=dict(run_id="run", tail=route),
                              read=AsyncMock(return_value=json.dumps(body).encode()))
        return asyncio.run(proxy.handle(req))
    (tmp_path/"run.json").write_text(json.dumps(spec))
    route = wire.PATHS[format]
    assert invoke("v1/unknown", orig).status == 404
    assert invoke(route, orig).status == 400
    raw = {**orig, field: orig[field] + [wire.user_item(format, "S")]}
    assert invoke(route, raw).status == 400
    app["specs"].clear()
    assert invoke(route, raw).status == 200
    logs = tmp_path/"logs/run"
    assert json.loads((logs/"first_request_parity.json").read_text())["ok"]
    assert json.loads((logs/"first_request_rewritten.json").read_text())["body"] == orig


def test_install_pins_and_raw_digest():
    for a in (CodexAdapter(), PiAdapter()):
        assert any(a.binary_sha256[a.version] in cmd for _, cmd, _ in a.install_commands())
        with pytest.raises(ValueError, match="unsupported"):
            a.install_commands("0")
    body = {"model": "x", "messages": []}
    ref = wire.body_reference("chat", body)
    ref["raw_sha256"] = hashlib.sha256(b'{"model":"x","messages":[]}').hexdigest()
    assert wire.compare_native(body, ref)["ok"]
    assert not wire.compare_native(dict(reversed(list(body.items()))), ref)["ok"]


@pytest.mark.parametrize("step", [4, 7])
def test_pi_recorded_native_session(tmp_path, step):
    fixture = Path(__file__).resolve().parents[1] / "examples/fixtures/pi-native-trial"
    capture = json.loads((fixture/f"resumed-step-{step}.json").read_text())
    b, meta = bundle.build(adapter=PiAdapter(), trial_dir=fixture, atif_step=step, out=tmp_path,
                           agent_timeout=3600, run_id=capture["run_id"])
    actual, _ = proxy.rewrite(capture["body"], json.loads((b/"proxy_spec.json").read_text()), first=True)
    req = tmp_path/"request.json"
    req.write_text(json.dumps(actual))
    assert parity.check_files(b, req)["ok"]
    assert equiv.check(b/"session_trunc.jsonl", meta["cut_line"], req)["equivalent"]
    actual["messages"][0]["content"] += "x"
    req.write_text(json.dumps(actual))
    report = parity.check_files(b, req)
    assert not report["ok"] and report["content_problems"][0] == "messages[0] digest differs"


def test_native_sandbox_commands_and_aux(tmp_path, monkeypatch):
    from agent_snapshot_resume.agents.codex import CodexSnapshotResumeOptions
    from agent_snapshot_resume.agents.pi import PiSnapshotResumeOptions
    from agent_snapshot_resume.agents.native import NativeResumeMixin
    from unittest.mock import Mock

    for options in (PiSnapshotResumeOptions, CodexSnapshotResumeOptions):
        assert {"resume_bundle", "upstream_fake", "parity_gate"} <= options.model_fields.keys()

    codex_trial(tmp_path, monkeypatch)
    adapter = CodexAdapter()
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=4, agent_timeout=60, out=tmp_path/"out")
    env, remote, args, cfg_path, config = adapter.resume_setup(meta, "http://127.0.0.1:4031/r/run")
    assert remote == adapter.replay_session_path(meta)
    assert env["CODEX_HOME"] in cfg_path and args[-2:] == ["thread", meta["sentinel"]]
    assert 'base_url = "http://127.0.0.1:4031/r/run/v1"' in config
    (tmp_path/"agent/model-catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="catalog SHA256"):
        adapter.collect_aux(adapter.find_session(tmp_path), adapter.load_trace(tmp_path, 60), 3, None, b/"aux")
    with pytest.raises(ValueError, match="invalid native session path"):
        adapter.replay_session_path(dict(meta, session_relpath="agent/sessions/../../else.jsonl"))

    agent = NativeResumeMixin()
    agent.adapter = adapter
    agent._bundle = Mock(return_value=(b, meta))
    agent._resume_logs = Mock(return_value=tmp_path)
    agent._start_proxies = AsyncMock()
    agent._verify_seeded = AsyncMock()
    agent.exec_as_agent = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="codex-cli 0.154.0"))
    agent.exec_as_root = AsyncMock()
    agent.environment_logs_dir = Path("/logs/agent")
    agent._OUTPUT_FILENAME = "codex.txt"
    environment = SimpleNamespace(upload_file=AsyncMock(), default_user="agent")
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("", environment, None))
    seeded = agent._verify_seeded.await_args.args[1]
    assert seeded[remote].read_bytes() == (b/"session_trunc.jsonl").read_bytes()
    assert environment.upload_file.await_count == 3
    assert seeded["/logs/agent/model-catalog.json"] == b / "aux/model-catalog.json"
    assert "timeout -s TERM" in agent.exec_as_agent.await_args.kwargs["command"]
    agent.exec_as_agent.return_value.stdout = ""
    with pytest.raises(RuntimeError, match="version differs"):
        asyncio.run(agent.install(environment))


@pytest.mark.parametrize("text,expected", [
    ("a" * 48_000, "a" * 48_000),
    ("a" * 48_001, "a" * 24_000 + "…1 tokens truncated…" + "a" * 24_000),
    ("a" * 48_005, "a" * 24_000 + "…2 tokens truncated…" + "a" * 24_000),
    ("a" * 23_999 + "中🙂" + "b" * 23_999, "a" * 23_999 + "…2 tokens truncated…" + "b" * 23_999),
    ("🙂" * 12_001, "🙂" * 6000 + "…1 tokens truncated…" + "🙂" * 6000),
])
def test_codex_truncates_tool_outputs(text, expected):
    from agent_snapshot_resume.agents.codex import project_item
    for typ in ("function_call_output", "custom_tool_call_output"):
        item = dict(type=typ, call_id="c", output=text)
        assert project_item(item, 10000)["output"] == expected and item["output"] == text
        blocks = [{"type": "input_text", "text": text}]
        assert project_item(dict(item, output=blocks), 10000)["output"][0]["text"] == expected
    assert project_item(dict(item, type="message"), 10000)["output"] == text


def test_codex_structured_outputs_share_one_token_budget():
    from agent_snapshot_resume.agents.codex import render_output
    def block(text):
        return {"type": "input_text", "text": text}
    image = {"type": "input_image", "image_url": "synthetic"}
    output = [block("ab"), block("x" * 47998), block("y"), image, block(""), block("z")]
    assert render_output(output, 10000) == [
        block("ab"), block("x" * 23998 + "…1 tokens truncated…" + "x" * 23998),
        image, block("[omitted 2 text items ...]"),
    ]


def test_codex_requires_pinned_catalog_at_build_and_resume(tmp_path, monkeypatch):
    from agent_snapshot_resume.agents import codex
    codex_trial(tmp_path, monkeypatch)
    exported = tmp_path / "agent/model-catalog.json"
    cached = tmp_path / "cache.json"
    exported.rename(cached)
    adapter = CodexAdapter(catalog_path=cached)
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=4, agent_timeout=60, out=tmp_path/"out")
    seed = b / "aux/model-catalog.json"
    assert seed.read_bytes() == cached.read_bytes()
    assert 'model_catalog_json = "/logs/agent/model-catalog.json"' in adapter.resume_setup(meta, "unused")[-1]
    exported.write_bytes(b"changed")
    with pytest.raises(ValueError, match="catalog SHA256"):
        adapter.trial_catalog(exported.parent, meta["native_model"])
    exported.unlink()
    with pytest.raises(ValueError, match="no supported token policy"):
        codex.read_catalog(cached, "unknown")
    cached.unlink()
    with pytest.raises(ValueError, match="missing pinned"):
        bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=4, agent_timeout=60, out=tmp_path/"absent")
    with pytest.raises(ValueError, match="requires a bundled"):
        adapter.resume_setup({**meta, "aux_counts": {}}, "unused")
    adapter.sanity_check_bundle(b, meta)
    seed.write_bytes(b"{}")
    with pytest.raises(ValueError, match="catalog SHA256"):
        adapter.sanity_check_bundle(b, meta)
    seed.unlink()
    with pytest.raises(ValueError, match="missing pinned"):
        adapter.sanity_check_bundle(b, meta)


def test_codex_artifact_content_position_and_continuations(monkeypatch):
    import copy
    def msg(role, *texts):
        return {"type": "message", "role": role, "id": "msg_01234567-89ab-cdef-0123-456789abcdef",
                "content": [{"type": "input_text", "text": text} for text in texts]}
    monkeypatch.setattr(wire, "SKILLS_SHA", wire.text_hash("skills `{CODEX_HOME}/skills/.system`"))
    monkeypatch.setattr(wire, "PERMISSIONS_SHA", wire.text_hash("fixture permissions"))
    dev = msg("developer", "skills `/logs/agent/skills/.system`", "fixture permissions")
    env = msg("user", "<environment_context>\n  <cwd>/task</cwd>\n  <shell>bash</shell>\n"
              "  <current_date>2026-11-19</current_date>\n  <timezone>Asia/Shanghai</timezone>\n"
              '  <filesystem><workspace_roots><root>/task</root></workspace_roots><permission_profile type="managed">'
              '<file_system type="restricted"><entry access="read"><special>:root</special></entry></file_system>'
              "</permission_profile></filesystem>\n</environment_context>")
    spec = dict(wire="responses", mode="hint", hint="hint", sentinel="unique",
                resume_artifacts=CodexAdapter().bundle_artifacts("/task"))
    sentinel = msg("user", "unique")
    orig = [dev, env, msg("user", "original task")]
    body = {"input": orig + [dev, env, sentinel]}
    spec["resume_artifacts"]["prefix_items"] = len(orig)
    with pytest.raises(ValueError, match="referenced cut"):
        wire.rewrite_native({"input": [dev, env, sentinel]}, spec, first=True)
    rewritten, actions = wire.rewrite_native(body, spec, first=True)
    assert rewritten["input"] == orig + [wire.user_item("responses", "hint")]
    assert actions == ["drop_resume_developer", "drop_resume_environment", "replace_sentinel"]
    later = {"input": body["input"] + [msg("user", "more")]}
    assert wire.rewrite_native(later, spec)[0]["input"] == rewritten["input"] + later["input"][-1:]
    malformed = [[env, dev, sentinel], [env, sentinel], [dev, env, sentinel, msg("user", "after")],
                 [dict(dev, extra=True), env, sentinel], [dev, env, dict(sentinel, id="invalid")]]
    for index, block in ((0, 0), (0, 1), (1, 0)):
        items = copy.deepcopy([dev, env, sentinel])
        items[index]["content"][block]["text"] += " unexpected"
        malformed.append(items)
    for items in malformed:
        with pytest.raises(ValueError):
            wire.rewrite_native({"input": orig + items}, spec, first=True)
    with pytest.raises(ValueError, match="cwd"):
        wire.rewrite_native(body, {**spec, "resume_artifacts": CodexAdapter().bundle_artifacts("/else")}, first=True)


@pytest.mark.parametrize("stream", [True, False])
def test_opt_in_responses_reference_end_to_end(tmp_path, monkeypatch, stream):
    from agent_snapshot_resume.proxy import passthrough_proxy as pp
    from unittest.mock import MagicMock
    codex_trial(tmp_path, monkeypatch)
    body = {"model": "m", "input": [{"type": "function_call_output", "call_id": "c", "output": "private result"},
                                    wire.user_item("responses", "private prompt")],
            "tools": [{"name": "private_tool"}], "instructions": "private instructions",
            "client_metadata": {"thread_id": "thread"}}
    raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    completed = {"status": "completed", "id": "r"}
    data = json.dumps({"type": "response.completed", "response": completed} if stream else completed).encode()
    payload = b"data: " + data + b"\r\n\r\n" if stream else data

    async def chunks():
        for i in range(0, len(payload), 3):
            yield payload[i:i+3]

    upstream = MagicMock(status=200, headers={"content-type": "text/event-stream" if stream else "application/json"})
    upstream.content.iter_any.side_effect = chunks
    sink = AsyncMock()
    monkeypatch.setattr(pp.web, "StreamResponse", lambda **kwargs: sink)
    app = dict(log_dir=tmp_path/"agent/model-proxy", port=1, proxy=None, upstream="unused", key="dummy",
               drop_fields=[], strip_is_error=False, strip_thinking=False, session=AsyncMock())
    app["session"].request.return_value = upstream
    req = SimpleNamespace(app=app, headers={}, method="POST", path="/v1/responses", path_qs="/v1/responses",
                          read=AsyncMock(return_value=raw))
    ir = CodexAdapter().load_trace(tmp_path, 60)
    log = lambda: next(f for f in app["log_dir"].glob("requests-*.jsonl") if f.name != "requests-1.jsonl")
    asyncio.run(pp.handle(req))
    assert "native_reference" not in json.loads(log().read_text())
    app["record_request_digests"] = True
    asyncio.run(pp.handle(req))
    assert app["session"].request.await_args.kwargs["data"] == raw
    assert b"".join(c.args[0] for c in sink.write.await_args_list) == payload * 2
    rec = json.loads(log().read_text().splitlines()[-1])
    assert "private" not in json.dumps(rec)
    content = rec["native_reference"]
    assert content["item_details"] == [
        dict(type="function_call_output", role=None, call_id="c", sha256=wire.digest(body["input"][0])),
        dict(type="message", role="user", call_id=None, sha256=wire.digest(body["input"][1]))]
    ref, error = CodexAdapter().native_reference(tmp_path, ir, 3, [])
    assert not error and ref["method"] == "captured-entry-id"
    assert wire.compare_native(body, ref["content"])["verdict"] == "ok"
    for changed in (dict(body, tools=[]), dict(body, instructions="drift"), dict(body, input=body["input"][::-1])):
        assert wire.compare_native(changed, ref["content"])["verdict"] == "mismatch"
    with log().open("a") as f:
        f.write(json.dumps(rec) + "\n")
    assert CodexAdapter().native_reference(tmp_path, ir, 3, [])[0] is None


def test_incomplete_responses_receipt():
    from agent_snapshot_resume.proxy.passthrough_proxy import ResponsesReceipt
    for data, sse in [(b'data: {"type":"response.created","response":{"id":"r"}}\n\n', True),
                      (b'data: {"type":"response.completed"}\n\n', True),
                      (b'{"status":"incomplete","id":"r"}', False), (b'{"status":"complet', False)]:
        tap = ResponsesReceipt(sse)
        tap.feed(data)
        assert tap.finish() is None


def test_projected_reference_includes_wire_ids_and_nulls():
    item = dict(type="reasoning", id="retained", summary=[], encrypted_content=None)
    row = {"type": "response_item", "payload": {**item, "internal_chat_message_metadata_passthrough": {"turn_id": 1}}}
    assert codex_history([row], 10000) == [item]
    ref = {"wire": "responses", "complete": False, "projection": "codex_rollout_v2", "fields": {},
           "items": [wire.digest(item)]}
    assert wire.compare_native({"input": [item]}, ref)["verdict"] == "incomplete_evidence"
    for bad in (dict(item, id="changed"), {k: v for k, v in item.items() if v is not None}):
        rep = wire.compare_native({"input": [bad]}, ref)
        assert rep["verdict"] == "mismatch" and rep["unequal_indices"] == [0]


def test_failed_or_duplicate_responses_event_has_no_receipt():
    from agent_snapshot_resume.proxy.passthrough_proxy import ResponsesReceipt
    done = b'data: {"type":"response.completed","response":{"id":"r","status":"completed"}}\n\n'
    for extra in (done, b'data: {"type":"response.failed"}\n\n'):
        tap = ResponsesReceipt(True)
        tap.feed(done + extra)
        assert tap.finish() is None
