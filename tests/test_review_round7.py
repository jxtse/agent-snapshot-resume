"""Native replay regressions from round seven, using only synthetic sessions."""
import asyncio
import copy
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_snapshot_resume import bundle
from agent_snapshot_resume.agents.codex import CodexAdapter, CodexSnapshotResume
from agent_snapshot_resume.proxy import runtime
from agent_snapshot_resume.replay import replay_inside as replay
from agent_snapshot_resume.sandbox import source
from test_native_adapters import codex_trial, pi_rows, write_lines
from test_review_native import pi_trial, replay_actions
from test_review_round3 import FULL

LOG = "/tmp/pi-bash-abcdef.log"


def pi_bash(root, cmd, text, error=False, details=None):
    rows = pi_rows()
    rows[0]["cwd"] = str(root)
    rows[2]["message"]["content"] = [dict(type="toolCall", name="bash", id="call", arguments={"command": cmd})]
    rows[3]["message"].update(isError=error, content=[dict(type="text", text=text)], details=details)
    return pi_trial(root, rows)


def codex_bash(root, monkeypatch, cmd, text, polled=False, repeated=False):
    rows, _ = codex_trial(root, monkeypatch)
    adapter = CodexAdapter()
    rows[1]["payload"].update(FULL, cwd=str(root))
    rows[3]["payload"]["arguments"] = json.dumps({"cmd": cmd})
    if polled:
        rows[4]["payload"]["output"] = rows[4]["payload"]["output"].replace("start", text[:1])
        rows[7]["payload"]["output"] = rows[7]["payload"]["output"].replace("done", text[1:])
    else:
        rows[4]["payload"]["output"] = "Chunk ID: x\nProcess exited with code 0\nOutput:\n" + text
        second = copy.deepcopy(rows[3:6]) if repeated else []
        if second:
            second[1]["payload"]["output"] = second[1]["payload"]["output"].replace("first", "second")
        rows = rows[:6] + second + rows[-3:]
    write_lines(adapter.find_session(root), rows)
    return adapter.load_trace(root, 60)


@pytest.mark.parametrize("name", ["codex", "pi"])
def test_native_env_used_and_bound(tmp_path, monkeypatch, name):
    cmd = 'printf "%s" "$DATA_ROOT" > marker'
    ir = pi_bash(tmp_path, cmd, "") if name == "pi" else codex_bash(tmp_path, monkeypatch, cmd, "")
    def resolve(value):
        cfg = {"agent": {"env": {"DATA_ROOT": value, "APP_TOKEN": "synthetic-secret"}}}
        (tmp_path/"config.json").write_text(json.dumps(cfg))
        return bundle.resolved_replay_plan(tmp_path, ir, 2, [])
    plan = resolve("fixture-a")
    monkeypatch.setenv("DATA_ROOT", "wrong-host")
    assert "synthetic-secret" not in json.dumps(plan)
    path, report = tmp_path/"plan.json", tmp_path/"report.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(sys, "argv", ["replay", str(path), str(report)])
    replay.main()
    assert (tmp_path/"marker").read_text() == "fixture-a"
    assert plan["agent_env_sha256"] == {"DATA_ROOT": hashlib.sha256(b"fixture-a").hexdigest()}
    assert resolve("fixture-b")["agent_env_sha256"] != plan["agent_env_sha256"]


@pytest.mark.parametrize("ref", ["$APP_TOKEN", "${LOCATION}", "${ROUTE:-none}", "$MISSING_API_KEY", "$UNRECORDED"])
def test_refuse_unrestorable_env(tmp_path, ref, monkeypatch):
    monkeypatch.setenv("UNRECORDED", "present-only-on-host")
    (tmp_path/"config.json").write_text(json.dumps({"agent": {"env": {
        "APP_TOKEN": "synthetic-secret", "LOCATION": "${UNRECORDED}", "ROUTE": "${APP_TOKEN}"}}}))
    ir = pi_bash(tmp_path, 'printf "%s" "' + ref + '" > marker', "")
    early = bundle.resolved_replay_plan(tmp_path, ir, 1, [])
    assert not early.get("replay_refusals")
    plan = bundle.resolved_replay_plan(tmp_path, ir, 2, [])
    assert any("call" in r and "env" in r for r in plan.get("replay_refusals", []))
    assert "synthetic-secret" not in json.dumps(plan)


@pytest.mark.parametrize("expression,blocked", [("data$sample", False), ("Sys.getenv('UNRECORDED')", True)])
def test_literal_interpreter_source(tmp_path, expression, blocked):
    cmd = "Rscript --vanilla - <<'RS'\n" + expression + "\nRS"
    ir = pi_bash(tmp_path, cmd, "")
    plan = bundle.resolved_replay_plan(tmp_path, ir, 2, [])
    assert bool(plan.get("replay_refusals")) == blocked


def test_indirect_env_name(tmp_path):
    (tmp_path/"config.json").write_text(json.dumps({"agent": {"env": {"SELECTOR": "DATA_ROOT"}}}))
    ir = pi_bash(tmp_path, 'printf "%s" "${!SELECTOR}" > marker', "")
    plan = bundle.resolved_replay_plan(tmp_path, ir, 2, [])
    assert "computed environment" in " ".join(plan.get("replay_refusals", []))


def test_aborted_bash(tmp_path):
    ir = pi_bash(tmp_path, "touch first; sleep 1; touch last", "partial\n\nCommand aborted", True)
    assert not bundle.plan_with_observations(ir, 1).get("replay_refusals")
    assert "abort" in " ".join(bundle.plan_with_observations(ir, 2).get("replay_refusals", [])).lower()


@pytest.mark.parametrize("cmd", ["cat input", "printf data > marker"])
def test_stdout_is_not_log_provenance(tmp_path, cmd):
    text = "sample\n\n[Showing lines 2-2001 of 2001. Full output: " + LOG + "]"
    ir = pi_bash(tmp_path, cmd, text)
    actions = bundle.plan_with_observations(ir, 2)["actions"]
    assert len(actions) == int(cmd.startswith("printf"))
    assert all("pi_output_file" not in a["args"] for a in actions)


@pytest.mark.parametrize("newline", [True, False])
def test_long_pi_line(tmp_path, monkeypatch, newline):
    raw = b"x"*60000 + (b"\n" if newline else b"")
    (tmp_path/"input").write_bytes(raw)
    size = "0B" if newline else "58.6KB"
    observed = "x"*51200 + f"\n\n[Showing last 50.0KB of line 1 (line is {size}). Full output: {LOG}]"
    ir = pi_bash(tmp_path, "cat input", observed, details={"fullOutputPath": LOG})
    plan = bundle.plan_with_observations(ir, 2)
    assert not plan.get("replay_refusals")
    monkeypatch.setattr(replay, "PI_LOG_DIR", str(tmp_path))
    result, = replay_actions(tmp_path, plan["actions"], monkeypatch)
    assert result["fidelity"] == "exact"
    assert result["pi_output_dimensions"] == {"totalBytes": len(raw), "totalLines": 1}


def test_large_log_is_streamed(tmp_path, monkeypatch):
    raw = (b"123456789\n"*400000)
    (tmp_path/"input").write_bytes(raw)
    text = ("123456789\n"*2000).rstrip("\n") + f"\n\n[Showing lines 398001-400000 of 400000. Full output: {LOG}]"
    ir = pi_bash(tmp_path, "cat input", text, details={"fullOutputPath": LOG})
    monkeypatch.setattr(replay, "PI_LOG_DIR", str(tmp_path))
    pread = replay.os.pread
    def bounded(fd, size, offset):
        assert size <= 1024*1024, "unbounded read"
        return pread(fd, size, offset)
    monkeypatch.setattr(replay.os, "pread", bounded)
    result, = replay_actions(tmp_path, bundle.plan_with_observations(ir, 2)["actions"], monkeypatch)
    assert result["fidelity"] == "exact"
    assert result["pi_output_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["pi_output_dimensions"] == dict(totalBytes=len(raw), totalLines=400000)
    assert (tmp_path/"report.json").stat().st_size < 150000


@pytest.mark.parametrize("polled", [False, True])
def test_codex_combined_stream(tmp_path, monkeypatch, polled):
    ir = codex_bash(tmp_path, monkeypatch, "printf a; printf b >&2; printf c; : > marker", "abc", polled)
    plan = bundle.plan_with_observations(ir, len(ir["steps"])-1)
    assert not plan.get("replay_refusals")
    result, = replay_actions(tmp_path, plan["actions"], monkeypatch)
    assert result["fidelity"] == "exact" and result["stdout_full"] == "abc"


def test_duplicate_cross_response(tmp_path, monkeypatch):
    ir = codex_bash(tmp_path, monkeypatch, "touch marker", "first", repeated=True)
    early = bundle.plan_with_observations(ir, 2)
    assert not early.get("replay_refusals") and early["actions"][0]["orig_text"] == "first"
    plan = bundle.plan_with_observations(ir, 3)
    assert "duplicate" in " ".join(plan.get("replay_refusals", [])).lower()


def test_codex_standalone(tmp_path, monkeypatch):
    # Substitute only the absolute guest runtime path when running on this host.
    standalone = str(tmp_path/"standalone")
    Path(standalone).symlink_to(sys.executable)
    monkeypatch.setattr(runtime, "RUNTIME", standalone)
    meta = {"agent": "codex", "fingerprint_roots": [str(tmp_path)]}
    command = source.fingerprint_command_for(meta)
    assert shlex.split(command)[0] == standalone
    subprocess.run(["/bin/sh", "-ec", command], env={"PATH": str(tmp_path)}, check=True, capture_output=True)
    sb = Mock()
    monkeypatch.setattr(source, "REPLAY_DIR", str(tmp_path))
    shutil.copyfile(replay.__file__, tmp_path/"replay_inside.py")
    (tmp_path/"plan.json").write_text('{"actions":[]}')
    for name in ("timeout", "mv"):
        (tmp_path/name).symlink_to(shutil.which(name))
    source.dispatch_replay(sb, str(tmp_path), 10, python=standalone)
    inner = shlex.split(sb.commands.run.call_args.args[0])[-1]
    subprocess.run(["/bin/bash", "-c", inner], env={"PATH": str(tmp_path)}, check=True)
    assert (tmp_path/"rc").read_text().strip() == "0"
    assert json.loads((tmp_path/"replay_report.json").read_bytes())["results"] == []
    agent = object.__new__(CodexSnapshotResume)
    agent._resume_logs = lambda: tmp_path
    async def local_exec(environment, command, env):
        assert shlex.split(command)[0] == standalone
        return SimpleNamespace(stdout=subprocess.check_output(["/bin/sh", "-ec", command], env=env))
    agent.exec_as_agent = local_exec
    seed = tmp_path/"seed"
    seed.write_text("fixture")
    asyncio.run(agent._verify_seeded(None, {str(seed): seed}, {"PATH": str(tmp_path)}))
