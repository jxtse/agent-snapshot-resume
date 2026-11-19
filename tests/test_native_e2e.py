import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_snapshot_resume.agents.pi import PiAdapter, REASONING_PROVIDER, pi_models, pi_args
from agent_snapshot_resume.proxy import native_wire as wire, passthrough_proxy as pp, resume_proxy, runtime
from agent_snapshot_resume.proxy.pi_receipts import bind

FIXTURE = Path(__file__).resolve().parents[1] / "examples/fixtures/pi-native-trial"


def partial(body):
    ref = wire.body_reference("responses", body)
    ref.update(complete=False, projection="codex_rollout_tokens", logged_item_count=len(ref["items"]),
               missing_evidence=["original_request_envelope"])
    ref["fields"] = {k: ref["fields"][k] for k in ("model", "instructions")}
    return {"content": ref}


def test_translator_and_incomplete_gate(tmp_path, monkeypatch):
    script = b"def build_chat_request(b):\n return {'messages': b['input']}, {}, {'dropped_fields':['client_metadata']}\n"
    checksum = hashlib.sha256(script).hexdigest()
    local = tmp_path / runtime.script_name(checksum)
    local.write_bytes(script)
    assert runtime.verified_script(local, checksum) == local
    with pytest.raises(ValueError):
        runtime.verified_script(local, "0"*64)
    with pytest.raises(ValueError):
        runtime.verified_script(local, None)
    command = runtime.passthrough_start_command("https://model.invalid", script_sha256=checksum)
    assert "--proxy ''" in command and local.name in command and "--key-file <(" in command
    monkeypatch.setattr(resume_proxy, "TRANSLATOR_DIR", tmp_path)
    translate = resume_proxy.load_translation_function(checksum)
    body = {"input": [wire.user_item("responses", "example")], "model": "m", "instructions": "i", "client_metadata": {}}
    ref = partial(body)
    ref["metadata"] = {"req_bytes": len(json.dumps(translate(body)[0]))}
    def judge(candidate, allow=True):
        return wire.evaluate_native_gate(candidate, ref, [], allow, translate)
    assert not judge(body, False)["admitted"]
    admitted = judge(body)
    assert admitted["verdict"] == "incomplete_evidence" and not admitted["ok"] and admitted["admitted"]
    assert judge(dict(body, client_metadata={"counter": 2}))["freeze_sha256"] == admitted["freeze_sha256"]
    assert judge(dict(body, tools=[]))["freeze_sha256"] != admitted["freeze_sha256"]
    assert judge(dict(body, instructions="changed"))["verdict"] == "mismatch"
    ref["metadata"]["req_bytes"] += 1
    assert judge(body)["verdict"] == "mismatch" and not judge(body)["admitted"]
    ref["content"]["missing_evidence"].append("history")
    assert not judge(body)["admitted"]
    local.write_bytes(script + b"# changed")
    with pytest.raises(ValueError, match="digest"):
        resume_proxy.load_translation_function(checksum)


@pytest.mark.parametrize("stream", [False, True])
def test_record_chat_and_bind_by_response_id(tmp_path, monkeypatch, stream):
    capture = json.loads((FIXTURE/"resumed-step-4.json").read_text())["body"]
    raw = json.dumps(capture, ensure_ascii=False, separators=(",", ":")).encode()
    event = {"id": "r0", "choices": [{"index": 0, "finish_reason": "stop"}]}
    data = json.dumps(event).encode()
    payload = b"data: " + data + b"\r\n\r\ndata: [DONE]\r\n\r\n" if stream else data
    async def chunks():
        for i in range(0, len(payload), 3):
            yield payload[i:i+3]
    upstream = MagicMock(status=200, headers={"content-type": "text/event-stream" if stream else "application/json"})
    upstream.content.iter_any.side_effect = chunks
    sink = AsyncMock()
    monkeypatch.setattr(pp.web, "StreamResponse", lambda **kw: sink)
    app = dict(log_dir=tmp_path, port=1, proxy=None, upstream="unused", key="dummy",
               drop_fields=[], strip_is_error=False, strip_thinking=False, session=AsyncMock(), record_request_digests=True)
    app["session"].request.return_value = upstream
    path = "/v1/chat/completions"
    req = SimpleNamespace(app=app, headers={}, method="POST", path=path, path_qs=path, read=AsyncMock(return_value=raw))
    asyncio.run(pp.handle(req))
    assert app["session"].request.await_args.kwargs["data"] == raw
    assert b"".join(c.args[0] for c in sink.write.await_args_list) == payload
    receipt = json.loads(next(tmp_path.glob("requests-*.jsonl")).read_text())
    assert receipt["response_id"] == "r0" and wire.compare_native(capture, receipt["native_reference"])["ok"]
    session = [json.loads(l) for l in PiAdapter().find_session(FIXTURE).read_text().splitlines()]
    originals = [json.loads(l) for l in (FIXTURE/"agent/model-proxy/native-reference.jsonl").read_text().splitlines()]
    records = [dict(receipt, response_id=f"r{i}", native_reference=r["content"]) for i, r in enumerate(originals)]
    bound = bind(session, reversed(records))
    assert [{k: v for k, v in r.items() if k != "response_id"} for r in bound] == originals
    for incomplete in (records[:-1], [*records, records[0]]):
        with pytest.raises(ValueError):
            bind(session, incomplete)
    tap = pp.ChatReceipt(True)
    tap.feed(b"data: " + data + b"\n\n")
    assert tap.finish() is None


def test_pi_pinned_runtime_and_reasoning_options():
    steps = PiAdapter().install_commands()
    assert [name for name, _, _ in steps] == ["downloader", "node", "pi"]
    cfg = pi_models("http://127.0.0.1:4013/v1", "m", REASONING_PROVIDER)["providers"][REASONING_PROVIDER]
    model, = cfg["models"]
    assert model["reasoning"] and model["compat"]["requiresReasoningContentOnAssistantMessages"]
    assert model["compat"]["thinkingFormat"] == "deepseek" and model["thinkingLevelMap"]["max"] == "max"
    args = pi_args("session", "m", REASONING_PROVIDER, "task")
    assert args[args.index("--thinking")+1] == "max"


def test_required_gate_and_frozen_launch(tmp_path):
    from agent_snapshot_resume.bundle import build, bundle_identity
    from agent_snapshot_resume.launch import build_job_config
    from agent_snapshot_resume.agents.codex import CodexAdapter

    # Start with a real bundle directory, then supply a synthetic incomplete Responses reference.
    b, meta = build(adapter=PiAdapter(), trial_dir=FIXTURE, atif_step=4, out=tmp_path, agent_timeout=60)
    original = {"model": "fixture", "input": [wire.user_item("responses", "task")], "instructions": "fixture"}
    exp = partial(original)
    (b/"bundle.json").write_text(json.dumps(dict(meta, orig_request_at_k=exp)))
    (b/"proxy_spec.json").write_text(json.dumps(dict(wire="responses", mode="same", sentinel="S")))
    task = tmp_path/"task"
    task.mkdir()
    (task/"task.toml").write_text("[agent]\ntimeout_sec=60\n")
    kw = dict(adapter=CodexAdapter(), bundle=b, task_dir=task, jobs_dir=tmp_path, job_name="job",
              snapshot_id="fixture", parity_gate="require", fake=True)
    with pytest.raises(SystemExit, match="incomplete_evidence"):
        build_job_config(**kw)
    kw["allow_incomplete_evidence"] = True
    assert build_job_config(**kw)["agents"][0]["kwargs"]["allow_incomplete_evidence"]
    agent = tmp_path/"trial/agent"
    (agent/"resume").mkdir(parents=True)
    (agent/"resume/bundle_identity.json").write_text(json.dumps(bundle_identity(b)))
    spec = dict(upstream="fake", wire="responses", mode="same", sentinel="S",
                first_request_parity=dict(exp, route=resume_proxy.route_from_passthrough_args([])),
                allow_incomplete_evidence=True)
    (tmp_path/"run.json").write_text(json.dumps(spec))
    app = dict(spec_dir=tmp_path, log_dir=agent/"resume-proxy", counts={}, specs={})
    body = dict(original, input=original["input"]+[wire.user_item("responses", "S")])
    def invoke():
        request = SimpleNamespace(app=app, method="POST", headers={}, match_info=dict(run_id="run", tail="v1/responses"),
                                  read=AsyncMock(return_value=json.dumps(body).encode()))
        return asyncio.run(resume_proxy.handle(request))
    assert invoke().status == 200
    logs = agent/"resume-proxy/run"
    report = json.loads((logs/"first_request_parity.json").read_text())
    assert report["allow_incomplete_evidence"] and report["admitted"] and not report["ok"]
    kw.update(fake=False, upstream_url="https://example.invalid", freeze_first_request=logs/"first_request_rewritten.json")
    kwargs = build_job_config(**kw)["agents"][0]["kwargs"]
    assert kwargs["parity_verdict"] == "pending" and kwargs["fake_parity_verdict"] == "incomplete_evidence"
    assert kwargs["expected_first_request_sha256"] == wire.digest(original)
    body["model"] = "changed"
    app.update(specs={}, counts={})
    assert invoke().status == 400
    assert json.loads((logs/"first_request_parity.json").read_text())["verdict"] == "mismatch"
    with pytest.raises(SystemExit):
        build_job_config(**kw)


def test_translator_remote_hash_failure_blocks_start(tmp_path):
    from agent_snapshot_resume.agents.base import SnapshotResumeAgentMixin
    script = tmp_path / "custom.py"
    script.write_bytes(b"# fixture\n")
    checksum = hashlib.sha256(script.read_bytes()).hexdigest()
    (tmp_path/"proxy_spec.json").write_text("{}")
    agent = SnapshotResumeAgentMixin()
    agent.logs_dir = tmp_path
    agent.options = SimpleNamespace(upstream_fake=True, passthrough_script=script, passthrough_script_sha256=checksum)
    agent.exec_as_root = AsyncMock(return_value=SimpleNamespace(return_code=1))
    env = SimpleNamespace(upload_file=AsyncMock())
    with pytest.raises(RuntimeError, match="sandbox.*SHA-256"):
        asyncio.run(agent._start_proxies(env, tmp_path, {}))
    env.upload_file.assert_awaited_once_with(script, f"{runtime.PROXY_DIR}/src/{runtime.script_name(checksum)}")
    assert "sha256sum -c -" in agent.exec_as_root.await_args.kwargs["command"]
    assert agent.exec_as_root.await_count == 1


def test_control_only_prep_and_status(tmp_path, monkeypatch, capsys):
    import csv
    from agent_snapshot_resume.batch import Batch
    from agent_snapshot_resume.cli import main
    manifest = tmp_path/"manifest.csv"
    task = tmp_path/"tasks/fixture-task"
    task.mkdir(parents=True)
    (task/"task.toml").write_text("[agent]\ntimeout_sec=1000\n")
    ir = PiAdapter().load_trace(FIXTURE, 1000)
    cut = next(s["line_range"][0] for s in ir["steps"] if s["atif_step_id"] == 7)
    with manifest.open("w") as out:
        writer = csv.writer(out)
        writer.writerows([["task", "cut_line", "source_dir"], [task.name, cut, FIXTURE]])
    batch = Batch(name="example", work_dir=tmp_path, manifest=manifest, tasks_dir=task.parent,
                  jobs_dir=tmp_path/"jobs", runs_dir=tmp_path/"runs", state_dir=tmp_path/"state",
                  agent="pi", hinted_reps=0, control_reps=1, cut_window=0)
    reached = []
    def boundary(row):
        reached.append("template")
        raise RuntimeError("offline sandbox boundary")
    monkeypatch.setattr(batch, "stage_template", boundary)
    batch.prep(batch.rows[0])
    assert reached == ["template"]
    record = json.loads((batch.unit_dir(task.name)/"bundles.json").read_text())
    assert set(record["bundles"]) == {"control"}
    assert record["bundles"]["control"]["parity_located"]
    batch.state.update(task.name, stage="ready", allow_incomplete_evidence=False, parity_verdict="ok")
    capsys.readouterr()
    assert main(["--env-file", "/dev/null", "batch", "status", "--name", "example",
                 "--work-dir", str(tmp_path), "--json"]) == 0
    row, = json.loads(capsys.readouterr().out)
    assert (row["parity_verdict"], row["allow_incomplete_evidence"]) == ("ok", False)


def test_codex_restores_only_identified_permissions(tmp_path):
    from agent_snapshot_resume.agents.codex import CodexAdapter, recorded_sandbox_mode
    context = dict(sandbox_policy={"type": "danger-full-access"}, approval_policy="never",
                   permission_profile={"type": "disabled"})
    mode = recorded_sandbox_mode(context)
    assert mode == "danger-full-access"
    assert recorded_sandbox_mode(dict(context, approval_policy="on-request")) == "unsupported"
    assert recorded_sandbox_mode(dict(context, permission_profile={"type": "managed"})) == "unsupported"
    meta = dict(native_sandbox_mode=mode, session_relpath="agent/sessions/2026/01/01/rollout-demo.jsonl",
                session_id="example", native_model="fixture", native_provider="fixture",
                aux_counts={"model_catalog": 1}, sentinel="S")
    import tomllib
    cfg = tomllib.loads(CodexAdapter().resume_setup(meta, "http://127.0.0.1:4031/r/test")[-1])
    assert cfg["sandbox_mode"] == mode and cfg["approval_policy"] == "never"
    with pytest.raises(ValueError, match="sandbox policy"):
        CodexAdapter().resume_setup(dict(meta, native_sandbox_mode="unsupported"), "unused")


def test_responses_fake_completes_one_assistant_message():
    reply = resume_proxy.native_fake_reply("responses", {"model": "fixture", "stream": True})
    events = [json.loads(line[6:]) for line in reply.text.splitlines() if line.startswith("data: ")]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created" and events[-1]["type"] == "response.completed"
    item, = events[-1]["response"]["output"]
    assert item == next(e["item"] for e in events if e["type"] == "response.output_item.done")
    assert item["role"] == "assistant" and item["content"][0]["text"] == "OK"


def test_harbor_baseline_install_and_run_commands(tmp_path, monkeypatch):
    import subprocess
    from agent_snapshot_resume import config
    from agent_snapshot_resume.agents.pi import PiBaseline
    monkeypatch.setattr(config, "upstream_url", lambda: "https://upstream.invalid")
    monkeypatch.setattr(config, "upstream_key", lambda: "fixture-process-only")
    env = SimpleNamespace(upload_file=AsyncMock())
    agent = SimpleNamespace(_version="1.4.0", logs_dir=tmp_path, model_name="asr-reasoning/fixture",
                            exec_as_root=AsyncMock(), exec_as_agent=AsyncMock(), _upload_config_text=AsyncMock(),
                            _OUTPUT_FILENAME="pi.txt")
    asyncio.run(PiBaseline.install(agent, env))
    asyncio.run(PiBaseline.run(agent, "exercise the fixture", env, None))
    root_calls = agent.exec_as_root.await_args_list
    agent_calls = agent.exec_as_agent.await_args_list
    digest_starts = [c for c in root_calls if "--record-request-digests" in c.kwargs["command"]]
    assert len(digest_starts) == 1 and digest_starts[0].kwargs["env"] == {runtime.KEY_ENV_IN_SANDBOX: "fixture-process-only"}
    for call in [*root_calls, *agent_calls]:
        command = call.kwargs["command"]
        subprocess.run(["bash", "-n"], input=command, text=True, check=True)
        assert "fixture-process-only" not in command
    assert all(runtime.KEY_ENV_IN_SANDBOX not in c.kwargs.get("env", {}) for c in agent_calls)
    assert "pi_receipts.py" in root_calls[-1].kwargs["command"]
    assert "--thinking max" in agent_calls[-1].kwargs["command"] and "baseline.jsonl" in agent_calls[-1].kwargs["command"]
    models = json.loads(agent._upload_config_text.await_args.kwargs["content"])
    assert models["providers"][REASONING_PROVIDER]["baseUrl"] == "http://127.0.0.1:4013/v1"



def test_codex_environment_context_without_cwd_when_resumed_in_recorded_cwd(monkeypatch):
    """Codex 0.154.0 on Linux omits <cwd>/<shell> when the resumed cwd equals the recorded one."""
    import pytest
    from agent_snapshot_resume.proxy import native_wire as nw
    skills_text = "skills root `/logs/agent/skills/.system` end"
    monkeypatch.setattr(nw, "SKILLS_SHA", nw.text_hash(skills_text.replace("/logs/agent", "{CODEX_HOME}")))
    mid = lambda n: f"msg_0000000{n}-0000-0000-0000-000000000000"
    developer = {"type": "message", "id": mid(1), "role": "developer",
                 "content": [{"type": "input_text", "text": skills_text}]}
    env = ('<environment_context>\n  <current_date>2026-11-19</current_date>\n  <timezone>Etc/UTC</timezone>\n'
           '  <filesystem><workspace_roots><root>/root/project</root></workspace_roots><permission_profile '
           'type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>\n'
           '</environment_context>')
    user = lambda text: {"type": "message", "id": mid(2), "role": "user",
                         "content": [{"type": "input_text", "text": text}]}
    arts = {"sandbox_mode": "danger-full-access", "codex_home": "/logs/agent", "cwd": "/root/project"}
    nw.check_codex_context(developer, user(env), arts)  # accepted
    with pytest.raises(ValueError):  # another workspace root is refused
        nw.check_codex_context(developer, user(env.replace("/root/project", "/tmp/elsewhere")), arts)
    with pytest.raises(ValueError):  # only one of the two omitted tags present is refused
        nw.check_codex_context(developer, user(env.replace("<current_date>", "<shell>bash</shell>\n  <current_date>")),
                               arts)


def test_chat_receipt_survives_client_hangup_after_final_event(tmp_path):
    """Pi closes the connection right after the last SSE event; the receipt must still bind the response id."""
    import asyncio, json as _json
    import aiohttp
    from aiohttp import web
    from agent_snapshot_resume.proxy import passthrough_proxy as pp

    chunks = [b'data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n',
              b'data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
              b'data: [DONE]\n\n']

    async def upstream(request):
        await request.read()
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        for c in chunks:
            await resp.write(c)
        await resp.write_eof()
        return resp

    async def main():
        up = web.Application(); up.router.add_post("/v1/chat/completions", upstream)
        r1 = web.AppRunner(up); await r1.setup(); s1 = web.TCPSite(r1, "127.0.0.1", 0); await s1.start()
        port = s1._server.sockets[0].getsockname()[1]
        app = web.Application()
        app.update(record_request_digests=True, key="k", strip_is_error=False, strip_thinking=False, drop_fields=[],
                   retry_403_marker=b"", upstream=f"http://127.0.0.1:{port}", proxy=None, port=0, cafile=None,
                   log_dir=str(tmp_path), keep_thinking_text=False)
        app.on_startup.append(pp.on_startup); app.on_cleanup.append(pp.on_cleanup)
        app.router.add_route("*", "/{tail:.*}", pp.handle)
        r2 = web.AppRunner(app); await r2.setup(); s2 = web.TCPSite(r2, "127.0.0.1", 0); await s2.start()
        pport = s2._server.sockets[0].getsockname()[1]
        body = _json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True})
        reader, writer = await asyncio.open_connection("127.0.0.1", pport)
        writer.write((f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {len(body)}\r\n\r\n{body}").encode())
        await writer.drain()
        got = b""
        while b"[DONE]" not in got:
            got += await reader.read(4096)
        writer.transport.abort()  # hang up before the chunked terminator
        await asyncio.sleep(0.3)
        await r2.cleanup(); await r1.cleanup()

    asyncio.run(main())
    rows = [_json.loads(l) for f in tmp_path.glob("requests-*.jsonl") for l in f.read_text().splitlines()]
    assert len(rows) == 1, rows
    assert rows[0]["status"] == 200 and rows[0]["response_id"] == "cmpl-1"


def test_pi_bash_truncation_rendering_matches_pi_1_4_0():
    """Replay compares a long Pi Bash result against Pi's own tail view (2000 lines / 50 KiB + footer)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/agent_snapshot_resume/replay"))
    import replay_inside as ri
    full = "".join(f"line {i:05d} " + "x" * 20 + "\n" for i in range(4000))
    lines = full.rstrip("\n").split("\n")
    kept = []
    size = 0
    for line in reversed(lines):
        b = len(line) + (1 if kept else 0)
        if size + b > 50 * 1024 or len(kept) >= 2000:
            break
        kept.insert(0, line); size += b
    footer = f"\n\n[Showing lines {4000 - len(kept) + 1}-4000 of 4000 (50.0KB limit). Full output: /tmp/pi-bash-0123456789abcdef.log]"
    orig = "\n".join(kept) + footer
    assert ri.compare(orig, ri.pi_truncated_view(orig, full), 0) == "exact"
    assert ri.pi_truncated_view("plain", full) == full  # untouched without Pi's footer
    assert ri.compare(orig, ri.pi_truncated_view(orig, full.replace("line 03999", "line 0399X")), 0) != "exact"


def test_pi_steps_follow_harbor_atif_ids(tmp_path):
    """With agent/trajectory.json, Pi step ids come from the export, never from position (Codex review P1)."""
    import json as _json
    import shutil
    import pytest
    from agent_snapshot_resume.agents.pi import PiAdapter
    fixture = Path(__file__).resolve().parents[1] / "examples/fixtures/pi-native-trial"
    trial = tmp_path / "trial"
    shutil.copytree(fixture, trial)
    a = PiAdapter()
    positional = a.load_trace(trial, 3600)
    assert "atif_missing_step_ids_positional" in positional["flags"]
    steps = positional["steps"]
    agent = [s for s in steps if s["source"] == "agent"]
    # Harbor export that drops the first agent step and numbers the rest non-positionally.
    exported = []
    for n, s in enumerate(agent[1:], start=100):
        calls = [dict(tool_call_id=e["call_id"]) for e in s["events"] if e["type"] == "tool_call"]
        exported.append(dict(step_id=n, source="agent", timestamp=s["timestamp"], tool_calls=calls))
    (trial / "agent/trajectory.json").write_text(_json.dumps({"steps": exported}))
    ir = a.load_trace(trial, 3600)
    got = {s["line_range"][0]: s["atif_step_id"] for s in ir["steps"] if s["source"] == "agent"}
    assert got[agent[0]["line_range"][0]] is None  # dropped by the export: refused, not guessed
    for n, s in enumerate(agent[1:], start=100):
        assert got[s["line_range"][0]] == n
    assert any(f.startswith("atif_unbound_steps:") for f in ir["flags"])
    assert "atif_missing_step_ids_positional" not in ir["flags"]
