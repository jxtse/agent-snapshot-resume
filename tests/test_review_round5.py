"""Regression checks for the fifth native adapter review."""
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_snapshot_resume import bundle, launch
from agent_snapshot_resume.agents.base import seed_hash_command
from agent_snapshot_resume.agents.codex import CodexAdapter
from agent_snapshot_resume.agents.pi import PiAdapter
from agent_snapshot_resume.proxy import resume_proxy
from agent_snapshot_resume.replay import replay_inside
from agent_snapshot_resume.sandbox import source
from agent_snapshot_resume.trace.ir import bash_is_read_only
from test_native_adapters import codex_trial, pi_rows, write_lines
from test_review_native import bash_plan, pi_trial, replay_actions
from test_review_round3 import FULL
from test_review_round4 import EXIT, refused


@pytest.mark.parametrize("name", ["pi", "codex"])
def test_native_job_env(tmp_path, monkeypatch, name):
    adapter = PiAdapter() if name == "pi" else CodexAdapter()
    if name == "pi":
        pi_trial(tmp_path, pi_rows())
    else:
        codex_trial(tmp_path, monkeypatch)
    (tmp_path/"task.toml").write_text("[agent]\n")
    env = dict(DATA_ROOT="/task/data", APP_TOKEN="synthetic", SECOND="${APP_TOKEN}",
               OPENAI_BASE_URL="old", https_proxy="old", ASR_UPSTREAM_URL="old")
    monkeypatch.setenv("APP_TOKEN", "synthetic")
    (tmp_path/"config.json").write_text(json.dumps({"agent": {"env": env, "kwargs": {"obsolete": True}}}))
    dest, _ = bundle.build(adapter=adapter, trial_dir=tmp_path, agent_timeout=60,
                           atif_step=3, out=tmp_path/"out")
    cfg = launch.build_job_config(adapter=adapter, bundle=dest, task_dir=tmp_path, job_name="j", jobs_dir=tmp_path,
                                  original_trial_dir=str(tmp_path), snapshot_id="fixture", fake=True)
    agent, = cfg["agents"]
    assert agent["env"] == {"DATA_ROOT": "/task/data"}
    assert "obsolete" not in agent["kwargs"]


@pytest.mark.parametrize("cmd", ["cp /logs/agent/sessions/2026/01/01/rollout-demo.jsonl copy",
                                 'cat "${CODEX_HOME}"/sessions/*/*/*/*.jsonl > copy',
                                 "cp ~/.codex/sessions/2026/01/01/*.jsonl copy"])
def test_codex_session_read(tmp_path, monkeypatch, cmd):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    rows[3]["payload"]["arguments"] = json.dumps({"cmd": cmd})
    rows[4]["payload"]["output"] = EXIT
    adapter = CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows[:6]+[rows[-2]])
    ir = adapter.load_trace(tmp_path, 60)
    assert not bundle.plan_with_observations(ir, 1).get("replay_refusals")
    refused(bundle.plan_with_observations(ir, 2), "session")


def test_codex_request_reverse_uniqueness(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    adapter = CodexAdapter()
    path = tmp_path/"agent/model-proxy/requests-1.jsonl"
    rec = json.loads(path.read_bytes())
    write_lines(path, [dict(rec, ts=1767225601)])
    rows[-3]["timestamp"] = "2026-01-01T00:00:05Z"
    second = dict(rows[-1], payload=dict(rows[-1]["payload"], response_id="second"))
    write_lines(adapter.find_session(tmp_path), rows+[rows[-2], second])
    ir = adapter.load_trace(tmp_path, 60)
    for index in (-2, -1):
        assert "request_start_s" not in ir["steps"][index]
        assert adapter.native_reference(tmp_path, ir, index, [json.dumps(rows[0])])[0] is None
    write_lines(path, [dict(rec, response_id="r", ts=1767225601)])
    ir = adapter.load_trace(tmp_path, 60)
    assert "request_start_s" in ir["steps"][-2] and "request_start_s" not in ir["steps"][-1]


@pytest.mark.parametrize("available", [False, True])
def test_pi_bootstrap_python(tmp_path, available):
    # Minimal PATH with curl and a simulated offline package manager.
    if available:
        (tmp_path/"python3").symlink_to(sys.executable)
    (tmp_path/"curl").symlink_to("/usr/bin/true")
    apt = tmp_path/"apt-get"
    apt.write_text('#!/bin/sh\ncase "$1" in\nupdate) exit 0;;\ninstall) case " $* " in *" python3 "*) '
                   f'/bin/ln -s "{sys.executable}" "$PATH/python3";; *) exit 2;; esac;; esac\n')
    apt.chmod(0o755)
    setup = PiAdapter().install_commands()[0][1]
    subprocess.run(["/bin/sh", "-ec", setup], env={"PATH": str(tmp_path)}, check=True)
    assert (tmp_path/"python3").is_file()
    plan, report = tmp_path/"plan.json", tmp_path/"report.json"
    plan.write_text(json.dumps(dict(actions=[], initial_cwd=str(tmp_path))))
    minimal_env = {"PATH": str(tmp_path)}
    subprocess.run(["python3", replay_inside.__file__, plan, report], env=minimal_env, check=True, capture_output=True)
    assert json.loads(report.read_bytes())["results"] == []
    actual = subprocess.check_output(["/bin/sh", "-ec", seed_hash_command([str(plan)])], env=minimal_env)
    assert json.loads(actual) == {str(plan):hashlib.sha256(plan.read_bytes()).hexdigest()}


@pytest.mark.parametrize("cmd,expected", [("date", True), ("date '+[%F] %T'", True), ("date -u +%s", True),
                                          ("date -R", True), ("date -Iseconds", True), ("date -uR", True),
                                          ("date -s tomorrow", False), ("date 11011200", False),
                                          ("date -u 11011200", False), ("date +%s > output", False)])
def test_date_queries(cmd, expected):
    assert bash_is_read_only(cmd) is expected


@pytest.mark.parametrize("change", ["delete", "rename", "rewrite", "keep"])
def test_pi_cut_output_manifest(tmp_path, monkeypatch, change):
    recorded = "/tmp/pi-bash-0123456789abcdef.log"
    data = b"x\n" * 2001
    (tmp_path/"long").write_bytes(data)
    filename = Path(recorded).name
    obs = "\n".join(["x"]*2000) + "\n\n[Showing lines 2-2001 of 2001. Full output: "+recorded+"]"
    actions = bash_plan(tmp_path, "cat long", obs, details={"fullOutputPath": recorded})
    commands = {"delete": "rm "+filename, "rename": "mv "+filename+" moved", "rewrite": "printf new > "+filename}
    if change in commands:
        actions.append(dict(tool="Bash", args={"command": commands[change]}, cwd=str(tmp_path),
                            side_effect="exec", call_id="later", orig_text="", is_error=False))
    monkeypatch.setattr(replay_inside, "PI_LOG_DIR", str(tmp_path))
    replay_actions(tmp_path, actions, monkeypatch)
    report = json.loads((tmp_path/"report.json").read_bytes())
    assert report["results"][0]["pi_output_sha256"] == hashlib.sha256(data).hexdigest()
    sb = Mock()
    sb.files.read.side_effect = lambda path, **kw: (tmp_path/Path(path).name).read_bytes()
    state = {}
    source._collect_gate_evidence(sb, state, report, Mock())
    if change in ("delete", "rename"):
        assert state["auxiliary_hashes"] == {} and state["pi_output_manifest"][recorded] is None
    else:
        digest = hashlib.sha256(data if change == "keep" else b"new").hexdigest()
        assert state["auxiliary_hashes"] == {recorded: digest}
        assert state["pi_output_manifest"][recorded]["sha256"] == digest
    if change == "rename":
        assert (tmp_path/"moved").read_bytes() == data


@pytest.mark.parametrize("disconnect", ["write_eof", "write", None])
def test_resume_records_disconnect(tmp_path, monkeypatch, disconnect):
    async def chunks():
        yield b"data: [DONE]\n\n"
    up = Mock(status=200, headers={}, content=SimpleNamespace(iter_any=chunks))
    app = dict(spec_dir=str(tmp_path), specs={}, counts={}, log_dir=str(tmp_path), compact_seen={},
               session=SimpleNamespace(request=AsyncMock(return_value=up)))
    (tmp_path/"r.json").write_text(json.dumps(dict(wire="chat", upstream="http://fixture.invalid", sentinel="marker",
                                                  mode="same")))
    body = {"messages": [{"role":"user","content":"marker"}], "model":"fixture"}
    req = SimpleNamespace(app=app, match_info=dict(run_id="r", tail="v1/chat/completions"), method="POST",
                          read=AsyncMock(return_value=json.dumps(body).encode()), headers={}, query_string="")
    resp = SimpleNamespace(prepare=AsyncMock(), write=AsyncMock(), write_eof=AsyncMock())
    if disconnect:
        getattr(resp, disconnect).side_effect = ConnectionResetError()
    monkeypatch.setattr(resume_proxy.web, "StreamResponse", lambda **kwargs: resp)
    asyncio.run(resume_proxy.handle(req))
    line, = (tmp_path/"r/requests.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["status"] == 200 and row["n"] == 0 and row["resp_bytes"] == 14
    assert bool(row.get("client_closed_before_eof")) == bool(disconnect)
    up.release.assert_called_once()


def test_pi_edit_replacement_decoding(tmp_path):
    target = tmp_path/"data"
    target.write_bytes(b"\xef\xbb\xbfhello\r\n\xff\xe2\x82\r\n")
    assert replay_inside.edit_original_spans(target, [{"old_string":"hello", "new_string":"hi"}], False) == (
        "applied", "exact")
    assert target.read_bytes() == "\ufeffhi\r\n\ufffd\ufffd\r\n".encode()


def test_pi_resume_checks_cut_manifest(tmp_path, monkeypatch):
    from agent_snapshot_resume.agents.pi import PiSnapshotResume, PiSnapshotResumeOptions

    # Carry the source manifest through Harbor's normal launch config to the resume entry point.
    pi_trial(tmp_path, pi_rows())
    adapter = PiAdapter()
    dest, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, agent_timeout=60,
                              atif_step=3, out=tmp_path/"out")
    alive, gone = "/tmp/pi-bash-cafe.log", "/tmp/pi-bash-fade.log"
    plan_path = dest/"plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["actions"] = [dict(args={"pi_output_file": path}) for path in (alive, gone)]
    plan_path.write_text(json.dumps(plan))
    (tmp_path/"task.toml").write_text("[agent]\n")
    digest = hashlib.sha256(b"modified before cut").hexdigest()
    manifest = {alive: dict(sha256=digest, uid=0, gid=0, mode=0o640), gone: None}
    state = tmp_path/"state.json"
    state.write_text(json.dumps(dict(snapshot_id="fixture", replay_gate_passed=True, fingerprint_source="fixture",
                                     run_id=meta["run_id"], cut_identity=bundle.snapshot_cut_identity(dest),
                                     auxiliary_hashes={alive:digest}, pi_output_manifest=manifest)))
    cfg = launch.build_job_config(adapter=adapter, bundle=dest, task_dir=tmp_path, job_name="j",
                                  jobs_dir=tmp_path, snapshot_state=str(state), fake=True)
    assert cfg["environment"]["kwargs"]["expected_auxiliary_hashes"] == {alive:digest}
    options = PiSnapshotResumeOptions(**cfg["agents"][0]["kwargs"])
    agent = SimpleNamespace(adapter=adapter, _bundle=lambda: (dest, meta), options=options,
                            exec_as_agent=AsyncMock(), _resume_base_url=lambda run: "")
    monkeypatch.setattr(adapter, "sanity_check_bundle", lambda *args: None)

    class SeedReached(Exception):
        pass

    monkeypatch.setattr(adapter, "resume_setup", Mock(side_effect=SeedReached))
    with pytest.raises(SeedReached):
        asyncio.run(PiSnapshotResume.run(agent, "", None, None))
    keep, absent = [c.kwargs["command"] for c in agent.exec_as_agent.await_args_list]
    assert alive in keep and "0:0:640" in keep and digest in keep
    assert gone in absent and "! -e" in absent and "! -L" in absent
    options.pi_output_manifest = {alive: manifest[alive]}
    with pytest.raises(RuntimeError, match="manifest"):
        asyncio.run(PiSnapshotResume.run(agent, "", None, None))
