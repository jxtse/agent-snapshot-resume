"""Behavioral regressions for the second review of native harness support."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_snapshot_resume import bundle, launch, screen
from agent_snapshot_resume.agents import codex, pi
from agent_snapshot_resume.proxy import native_wire as wire, passthrough_proxy, pi_receipts, resume_proxy
from agent_snapshot_resume.replay import replay_inside as replay
from agent_snapshot_resume.replay.gate import validate_replay
from agent_snapshot_resume.trace.ir import bash_is_read_only
from test_native_adapters import codex_trial, pi_rows, write_lines
from test_review_native import pi_trial, replay_actions


@pytest.mark.parametrize("code", [0, 1])
def test_patch_output(tmp_path, monkeypatch, code):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    call = dict(type="custom_tool_call", name="apply_patch", call_id="p", input="test patch")
    answer = dict(type="custom_tool_call_output", call_id="p",
                  output=f"Exit code: {code}\nWall time: 0.2 seconds\nOutput:\nbody\n")
    rows = rows[:3] + [dict(type="response_item", timestamp=rows[0]["timestamp"], payload=p) for p in (call, answer)]
    adapter = codex.CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows)
    step = adapter.load_trace(tmp_path, 60)["steps"][-1]
    invoke, observation = step["events"]
    assert observation["text"] == ("Exit code 1\n" if code else "") + "body\n"
    assert invoke["is_error"] == bool(code) and invoke["duration_s"] == .2
    result = {"call_id": "p", "status": "ran", "fidelity": "exact", "rc": code + 2}
    for policy in ("strict", "lenient"):
        with pytest.raises(RuntimeError, match="codex_exit_code_changed"):
            validate_replay({"actions": [invoke]}, {"results": [result]}, 0, policy=policy)


@pytest.mark.parametrize("command", ["sort -o out in", "sort --output=out in", "uniq in out",
                                     "env touch out", "xxd -r in out", "echo $(touch out)"])
def test_writing_shell_retained(command):
    assert not bash_is_read_only(command)
    assert pi.tool_action("bash", {"command": command}, "/task")[2] == "exec"
    assert bash_is_read_only("ls -l && head input | sort -n")


PROMPTS = json.loads((Path(__file__).parent/"fixtures/pi-1.4.0-summary-prompts.json").read_bytes())


def summary(kind, transcript, previous="old"):
    if kind == "turn_prefix":
        text = f"# Conversation\n{transcript}\n\n# Instructions\n"
    else:
        text = f"<conversation>\n{transcript}\n</conversation>\n\n"
        if kind == "update":
            text += f"<previous-summary>\n{previous}\n</previous-summary>\n\n"
    return {"model": "recorded", "messages": [{"role": "system", "content": PROMPTS["system"]},
        {"role": "user", "content": text + PROMPTS[kind]}]}


def receipt(body, rid):
    ref, _ = passthrough_proxy.request_digests(json.dumps(body).encode(), "chat")
    return dict(path="/v1/chat/completions", status=200, response_id=rid, native_reference=ref)


def test_classify_pi_summary_receipts():
    rows = [dict(type="session", version=3, id="s"), dict(type="message", id="e", message=dict(
        role="assistant", responseId="a")), dict(type="compaction", id="compact", summary="reviewed")]
    normal = receipt({"messages": [wire.user_item("chat", "hi")]}, "a")
    aux = [receipt(summary(k, "[User]: history"), k) for k in ("summary", "turn_prefix")]
    assert [r["entry_id"] for r in pi_receipts.bind(rows, [normal, *aux])] == ["e"]
    for invalid in ([normal, normal, *aux], aux, [normal, *aux, dict(normal, response_id="unrelated")]):
        with pytest.raises(ValueError):
            pi_receipts.bind(rows, invalid)


@pytest.mark.parametrize("kind", ["summary", "update", "turn_prefix"])
@pytest.mark.parametrize("mode", ["same", "hint"])
def test_serialized_summary_marker(kind, mode):
    before, after = "[User]: history", "[Assistant]: response"
    marker = "ZZZ_RESUME_TEST"
    spec = dict(wire="chat", sentinel=marker, mode=mode, hint="next step")
    join = "\n\n".join
    body = summary(kind, join([before, "[User]: " + marker, after]))
    result, _ = wire.rewrite_native(body, spec)
    assert result == summary(kind, join([before] + (["[User]: next step"] if mode == "hint" else []) + [after]))
    for bad in (summary(kind, "[Tool result]: " + marker), summary(kind, "inline " + marker),
                summary(kind, join(["[User]: " + marker]*2))):
        with pytest.raises(ValueError):
            wire.rewrite_native(bad, spec)
    with pytest.raises(ValueError):
        wire.rewrite_native(body, spec, first=True)


def test_codex_budget_matches_selected_request(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    adapter = codex.CodexAdapter()
    rows[-3]["timestamp"] = "2026-01-01T00:00:20Z"
    write_lines(adapter.find_session(tmp_path), rows)
    path = tmp_path/"agent/model-proxy/requests-1.jsonl"
    rec = dict(json.loads(path.read_text()), ts="2026-01-01T00:00:04+0000")
    write_lines(path,[dict(rec, session="wrong", ts="2026-01-01T00:00:02Z"), rec])
    step = adapter.load_trace(tmp_path, 60)["steps"][-1]
    assert step["elapsed_s"] == 3 and step["remaining_s"] == 57 and step["timestamp"] == rows[-3]["timestamp"]


@pytest.mark.parametrize("wrong", ["totalLines", "totalBytes", "sha256"])
def test_truncated_history_checked(tmp_path, monkeypatch, wrong):
    rows = pi_rows()
    rows[0]["cwd"] = str(tmp_path)
    raw = b"x\n"*2001
    path = "/tmp/pi-bash-0123456789abcdef.log"
    (tmp_path/"input").write_bytes(raw)
    rows[2]["message"]["content"] = [dict(type="toolCall", name="bash", id="call", arguments=dict(command="cat input"))]
    rows[3]["message"].update(content=[dict(type="text", text="\n".join(["x"]*2000) +
        "\n\n[Showing lines 2-2001 of 2001. Full output: "+path+"]")], details=dict(
            fullOutputPath=path, truncation=dict(totalBytes=len(raw), totalLines=2001),
            fullOutputSha256=hashlib.sha256(raw).hexdigest()))
    details = rows[3]["message"]["details"]
    if wrong == "sha256":
        details["fullOutputSha256"] = "0"*64
    else:
        details["truncation"][wrong] += 1
    actions = bundle.plan_with_observations(pi_trial(tmp_path, rows), 2)["actions"]
    monkeypatch.setattr(replay, "PI_LOG_DIR", str(tmp_path))
    results = replay_actions(tmp_path, actions, monkeypatch)
    for policy in ("strict", "lenient"):
        with pytest.raises(RuntimeError, match="historical_output_mismatch"):
            validate_replay(dict(actions=actions), dict(results=results), 0, policy=policy)


@pytest.mark.parametrize("source", ["model", "ASR_MODEL"])
def test_native_override_rejected(tmp_path, monkeypatch, source):
    pi_trial(tmp_path, pi_rows())
    adapter = pi.PiAdapter()
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=3, agent_timeout=60, out=tmp_path/"out")
    (tmp_path/"task.toml").write_text("[agent]\n")
    kw = dict(adapter=adapter, bundle=b, task_dir=tmp_path, job_name="j", jobs_dir=tmp_path, snapshot_id="fixture", fake=True)
    if source == "model":
        kw["model"] = "wrong"
    else:
        monkeypatch.setenv(source, "wrong")
    with pytest.raises(ValueError, match="model override"):
        launch.build_job_config(**kw)


def test_codex_ambiguous_atif(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows.extend([rows[5], rows[-2].copy()])
    adapter = codex.CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows)
    (tmp_path/"agent/trajectory.json").write_text(json.dumps(dict(steps=[
        dict(step_id=9, source="agent", timestamp=rows[-1]["timestamp"])])))
    assert [st["atif_step_id"] for st in adapter.load_trace(tmp_path, 60)["steps"][-2:]] == [None, None]


def test_pi_log_owner_and_readback(tmp_path, monkeypatch):
    from agent_snapshot_resume.sandbox.source import dispatch_replay
    sb = Mock()
    dispatch_replay(sb, "/work", 10, pi_user="worker")
    cmd = sb.commands.run.call_args.args[0]
    assert "--pi-user worker" in cmd and cmd.startswith("/bin/bash -c")
    monkeypatch.setattr(replay, "PI_LOG_DIR", str(tmp_path))
    with (tmp_path/"source").open("w+b") as stream:
        stream.write(b"actual")
        stream.flush()
        result = replay.restore_pi_log("/tmp/pi-bash-abcd.log", stream, os.getuid())
    stat = (tmp_path/"pi-bash-abcd.log").stat()
    assert stat.st_mode & 0o777 == 0o600 and stat.st_uid == result["pi_output_reader_uid"] == os.getuid()


def test_pi_fuzzy_edit_preserves_lines(tmp_path):
    path = tmp_path/"x"
    data = "outside —  \r\nsmart “a” – x  \r\nuntouched É  \r\n"
    path.write_bytes(data.encode())
    edits = [dict(old_string='"a" - x',new_string="changed")]
    assert replay.edit_original_spans(path, edits, False) == ("applied","exact")
    assert path.read_bytes() == data.replace("“a” – x  ", "changed").encode()
    path.write_bytes((data + 'smart “a” – x\r\n').encode())
    assert replay.edit_original_spans(path, edits, False)[0] == "edit_failed"


def test_cache_binds_selected_receipt(tmp_path, monkeypatch):
    from test_batch import make_batch, fake_build
    batch = make_batch(tmp_path, agent="pi")
    batch._adapter = pi.PiAdapter()
    row = batch.rows[0]
    trial = Path(batch.source(row)["trial_dir"])
    pi_trial(trial, pi_rows())
    side = trial/"agent/model-proxy/native-reference.jsonl"
    ref = wire.body_reference("chat", {"messages":[wire.user_item("chat","one")]})
    receipt = dict(session_id="session",entry_id="a",content=ref)
    write_lines(side,[receipt])
    build = Mock(side_effect=fake_build)
    monkeypatch.setattr(bundle,"build",build)
    batch.stage_bundles(row, 5)
    build.reset_mock()
    batch.stage_bundles(row, 5)
    build.assert_not_called()
    write_lines(side,[dict(receipt, content=dict(ref, raw_sha256="a"*64))])
    batch.stage_bundles(row, 5)
    assert build.called


def test_screen_blocks_partial_reference(tmp_path, monkeypatch):
    codex_trial(tmp_path, monkeypatch)
    adapter = codex.CodexAdapter()
    ir = adapter.load_trace(tmp_path,60)
    rep, = screen.screen_trial(adapter,tmp_path,ir=ir,steps=[4])
    assert any(r["rule"] == "incomplete_evidence" and r["severity"] == "block" for r in rep["risks"])
    log = tmp_path/"agent/model-proxy/requests-1.jsonl"
    write_lines(log,[dict(json.loads(log.read_text()),n_items=1)])
    rep, = screen.screen_trial(adapter,tmp_path,ir=ir,steps=[4], allow_incomplete_evidence=True)
    assert all(r["rule"] != "incomplete_evidence" for r in rep["risks"])


def test_pi_shell_nonlogin(tmp_path, monkeypatch):
    # shopt tests the actual child shell mode independent of host startup scripts.
    action = dict(tool="Bash", side_effect="exec", call_id="shell", cwd=str(tmp_path),
                  args=dict(command="if shopt -q login_shell; then echo login; else echo nonlogin; fi",pi_bash=True))
    records = replay_actions(tmp_path, [action, dict(action,args={"command":action["args"]["command"]})], monkeypatch)
    assert [r["stdout_full"].strip() for r in records] == ["nonlogin","login"]


def test_native_hint_metrics(tmp_path):
    spec = dict(wire="chat",sentinel="ZZZ_MARK",mode="hint",hint="a hint",upstream="fake")
    app=dict(spec_dir=tmp_path, log_dir=tmp_path/"logs", counts={}, specs={})
    (tmp_path/"run.json").write_text(json.dumps(spec))
    for body in [{"model":"recorded", "messages":[wire.user_item("chat",spec["sentinel"])]}, summary("summary","[User]: ZZZ_MARK")]:
        request = SimpleNamespace(app=app, method="POST", headers={},match_info=dict(run_id="run",tail="v1/chat/completions"),
                                  read=AsyncMock(return_value=json.dumps(body).encode()))
        response = asyncio.run(resume_proxy.handle(request))
        assert response.status == 200
        logged = json.loads((tmp_path/"logs/run/requests.jsonl").read_text().splitlines()[-1])
        assert logged["hint_blocks"] == 1
    assert logged["compaction_request"] == "summary"


def test_native_status_unknown_without_rollout(tmp_path):
    from test_batch import make_batch
    b = make_batch(tmp_path, agent="pi")
    row = b.rows[0]
    trial = b.jobs_dir/"job"/"trial"
    trial.mkdir(parents=True)
    (trial/"result.json").write_text("{}")
    status = lambda: b.run_status(row, {}, "job", {"trial":"trial"})
    assert status()["n_compactions"] is None
    session = trial/"agent/pi/sessions/s.jsonl"
    write_lines(session, [dict(type="session"),dict(type="compaction")])
    prefix = session.read_bytes().splitlines(keepends=True)[0]
    meta = trial/"agent/resume/bundle.json"
    meta.parent.mkdir()
    meta.write_text(json.dumps(dict(agent="pi",session_relpath="agent/pi/sessions/s.jsonl",cut_line=1,
                                    prefix_sha256=hashlib.sha256(prefix).hexdigest())))
    assert status()["n_compactions"] == 1


def test_batch_screens_again_after_evidence_changes(tmp_path):
    from test_batch import make_batch
    driver = make_batch(tmp_path, agent="pi")
    driver._adapter = pi.PiAdapter()
    row = dict(driver.rows[0], cut_line=2)
    trial = Path(driver.source(row)["trial_dir"])
    pi_trial(trial, pi_rows())
    side = trial/"agent/model-proxy/native-reference.jsonl"
    ref = wire.body_reference("chat", {"messages":[wire.user_item("chat","history")]})
    write_lines(side,[dict(session_id="session", entry_id="a", content=ref)])
    assert driver.stage_screen(row)["cut_line"] == 5
    write_lines(side,[dict(session_id="session", entry_id="w", content=ref)])
    assert driver.stage_screen(row)["cut_line"] == 2
    side.unlink()
    with pytest.raises(RuntimeError, match="cut"):
        driver.stage_screen(row)


def test_fuzzy_uniqueness_for_exact_hit(tmp_path):
    p = tmp_path/"text"
    original = "a-b\na—b\n"
    p.write_text(original)
    assert replay.edit_original_spans(p,[dict(old_string="a-b", new_string="c")],False)[0] == "edit_failed"
    assert p.read_text() == original


def test_digest_only_compaction_receipt():
    session = [dict(type="session", version=3, id="s"),
               dict(id="r",message=dict(role="assistant",responseId="r")),dict(type="compaction")]
    normal = receipt({"messages":[wire.user_item("chat","normal")]},"r")
    old = receipt(summary("update","[User]: testing"), "c")
    old["native_reference"].pop("pi_compaction_request", None)
    assert len(pi_receipts.bind(session,[normal,old])) == 1
    with pytest.raises(ValueError):
        pi_receipts.bind(session[:-1],[normal,old])
