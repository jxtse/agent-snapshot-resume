"""Cut-local replay and terminal evidence from the third native review."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_snapshot_resume import bundle, screen
from agent_snapshot_resume.agents import codex, pi
from agent_snapshot_resume.agents.native import NativeResumeMixin
from agent_snapshot_resume.proxy import native_wire
from agent_snapshot_resume.replay.gate import validate_replay
from agent_snapshot_resume.sandbox import source
from test_batch import make_batch
from test_native_adapters import codex_trial, pi_rows, write_lines
from test_review_native import pi_trial, replay_actions


def plan(ir, k):
    return bundle.plan_with_observations(ir, k)


def assert_refused(plan, reason):
    assert screen.screen_plan(plan)["verdict"] == "block"
    with pytest.raises(RuntimeError, match=reason):
        validate_replay(plan, {"results": []}, 0)


@pytest.mark.parametrize("name", ["write", "edit"])
@pytest.mark.parametrize("unconfirmed", ["aborted", "wrong-id", "late"])
def test_pi_unconfirmed(tmp_path, name, unconfirmed):
    rows = pi_rows()
    call = rows[2]["message"]["content"][0]
    call.update(name=name, arguments=dict(path="missing", oldText="x", newText="y", content="x"))
    rows[2]["message"]["stopReason"] = "aborted"
    if unconfirmed == "aborted":
        del rows[3:5]
        rows[-1]["parentId"] = "w"
    elif unconfirmed == "wrong-id":
        rows[3]["message"]["toolCallId"] = "different"
    else:
        result = rows.pop(3)
        rows[-1]["parentId"] = "w"
        rows.remove(rows[-2])
        rows.append(dict(result, parentId="a"))
    ir = pi_trial(tmp_path, rows)
    assert not plan(ir, 1)["actions"]
    assert_refused(plan(ir, 2), "toolResult")


@pytest.mark.parametrize("error,classified", [(True, True), (False, False)])
def test_pi_invalid_arguments(tmp_path, error, classified):
    rows = pi_rows()
    rows[2]["message"]["content"][0]["arguments"] = {"content": "x"}
    rows[3]["message"].update(isError=error, toolName="write")
    ir = pi_trial(tmp_path, rows)
    assert not plan(ir, 1)["actions"]
    if classified:
        assert not plan(ir, 2)["actions"]
        validate_replay(plan(ir, 2), {"results": []}, 0)
    else:
        assert_refused(plan(ir, 2), "arguments")


def test_pi_cancelled(tmp_path):
    rows = pi_rows()
    rows[3]["message"] = dict(role="bashExecution", command="touch out", cancelled=True)
    rows[2]["message"]["content"] = [dict(type="text", text="ok")]
    ir = pi_trial(tmp_path, rows)
    assert not plan(ir, 1)["actions"]
    assert_refused(plan(ir, 3), "cancelled")


def test_pi_bash_environment(tmp_path, monkeypatch):
    rows = pi_rows()
    rows[0]["cwd"] = str(tmp_path)
    rows[2]["message"]["thinkingLevel"] = "low"
    call = rows[2]["message"]["content"][0]
    call.update(name="bash", arguments={"command":
        'printf "%s\\n" "$PI_SESSION_FILE" "$PI_SESSION_ID" "$PI_PROVIDER" "$PI_MODEL" "$PI_REASONING_LEVEL" > env.txt'})
    resolved = plan(pi_trial(tmp_path, rows), 2)
    assert not resolved.get("replay_refusals")
    replay_actions(tmp_path, resolved["actions"], monkeypatch)
    assert (tmp_path/"env.txt").read_text().splitlines() == [
        "/logs/agent/pi/sessions/session.jsonl", "session", "local", "fixture", "low"]
    del rows[2]["message"]["thinkingLevel"]
    assert_refused(plan(pi_trial(tmp_path, rows), 2), "PI_REASONING_LEVEL")
    call["arguments"]["command"] = 'echo "${PI_CUSTOM:-unknown}" > out'
    assert_refused(plan(pi_trial(tmp_path, rows), 2), "PI_CUSTOM")


@pytest.mark.parametrize("source_level,override", [("low", None), ("low", "medium"), (None, None)])
def test_pi_cut_thinking(tmp_path, source_level, override):
    rows = pi_rows()
    if source_level:
        rows.insert(1, dict(type="thinking_level_change", id="level", parentId=None,
                            timestamp=rows[0]["timestamp"], thinkingLevel=source_level))
        rows[2]["parentId"] = "level"
    for row in rows:
        if row.get("message", {}).get("role") == "assistant":
            row["message"]["provider"] = pi.REASONING_PROVIDER
    if override:
        rows[-1]["message"]["thinkingLevel"] = override
    pi_trial(tmp_path, rows)
    adapter = pi.PiAdapter()
    _, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=3, out=tmp_path/"out", agent_timeout=60)
    args = adapter.resume_setup(meta, "http://localhost")[2]
    actual = args[args.index("--thinking")+1] if "--thinking" in args else None
    assert actual == (override or source_level)


FULL = dict(sandbox_policy={"type": "danger-full-access"}, approval_policy="never",
            permission_profile={"type": "disabled"})


@pytest.mark.parametrize("policy", [{"type": "read-only"}, {"type": "workspace-write"}])
def test_codex_historical_policy(tmp_path, monkeypatch, policy):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"]["sandbox_policy"] = policy
    rows.insert(6, dict(rows[1], payload={**rows[1]["payload"], **FULL}))
    adapter = codex.CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows)
    ir = adapter.load_trace(tmp_path, 60)
    assert ir["steps"][-1]["native_sandbox_mode"] == "danger-full-access"
    assert not plan(ir, 1)["actions"]
    assert_refused(plan(ir, 3), "sandbox")


@pytest.mark.parametrize("commands,blocked", [
    (["echo a > x", "echo b > x"], True), (["cat x", "ls"], False), (["cat x", "echo b > x"], False)])
def test_codex_parallel(tmp_path, monkeypatch, commands, blocked):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    calls, results = [], []
    for i, cmd in enumerate(commands):
        calls.append(dict(rows[3], payload=dict(rows[3]["payload"], call_id=str(i), arguments=json.dumps(dict(cmd=cmd)))))
        results.append(dict(rows[4], payload=dict(rows[4]["payload"], call_id=str(i), output="Chunk ID: 1\nProcess exited with code 0\nOutput:\ndone")))
    adapter = codex.CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows[:3] + calls + results + [rows[5], rows[-2]])
    ir = adapter.load_trace(tmp_path, 60)
    assert not plan(ir, 1)["actions"]
    p = plan(ir, 2)
    if blocked:
        assert_refused(p, "parallel")
    else:
        assert screen.screen_plan(p)["verdict"] != "block"


def test_codex_legacy_usage(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    strong = rows[-1]
    usage = strong["payload"]["usage"]
    count = dict(rows[5], payload=dict(type="token_count", info=dict(last_token_usage=usage)))
    adapter = codex.CodexAdapter()
    path = adapter.find_session(tmp_path)
    write_lines(path, rows[:-1] + [count, count])
    ir = adapter.load_trace(tmp_path, 60)
    assert "usage" not in ir["steps"][0] and ir["steps"][-1]["usage"] == usage
    ref, _ = adapter.native_reference(tmp_path, ir, 3, [json.dumps(r) for r in rows[:10]])
    assert ref and ref["content"]["complete"] is False
    write_lines(path, rows + [dict(count, payload=dict(type="token_count", info=dict(last_token_usage={})))])
    assert adapter.load_trace(tmp_path, 60)["steps"][-1]["usage"] == usage
    write_lines(path, rows[:-1] + [count, dict(strong, payload=dict(strong["payload"], usage={"input_tokens": 99}))])
    assert adapter.load_trace(tmp_path, 60)["steps"][-1]["usage"] == {"input_tokens": 99}


@pytest.mark.parametrize("change", ["home", "budget"])
def test_pi_batch_cache(tmp_path, change):
    batch = make_batch(tmp_path, agent="pi", hinted_reps=0)
    batch._adapter = pi.PiAdapter()
    row = batch.rows[0]
    trial = Path(batch.source(row)["trial_dir"])
    rows = pi_rows()
    rows[2]["message"]["content"][0]["arguments"]["path"] = "~/file"
    home = trial/"agent/pi/home.txt"
    home.parent.mkdir(parents=True)
    home.write_text("/first")
    pi_trial(trial, rows)
    write_lines(trial/"agent/model-proxy/native-reference.jsonl", [dict(
        session_id="session", entry_id="a", content=native_wire.body_reference("chat", {"messages": []}))])
    old = batch.stage_bundles(row, 5)
    old_id = bundle.bundle_identity(Path(old["bundles"]["control"]["dir"]))
    if change == "home":
        home.write_text("/second")
    else:
        (batch.task_dir(row)/"task.toml").write_text("[agent]\ntimeout_sec=900")
    new = batch.stage_bundles(row, 5)
    assert new["rebuilt_for_input_change"]
    actual = new["bundles"]["control"]
    if change == "home":
        assert json.loads((Path(actual["dir"])/"plan.json").read_bytes())["actions"][0]["args"]["file_path"] == "/second/file"
    else:
        assert actual["agent_budget_s"] == old["bundles"]["control"]["agent_budget_s"] + 300
        assert bundle.bundle_identity_diff(bundle.bundle_identity(Path(actual["dir"])), old_id) == ["agent_budget_s"]
    assert not batch.stage_bundles(row, 5)["rebuilt_for_input_change"]


def test_batch_physical_lines(tmp_path, monkeypatch):
    batch = make_batch(tmp_path, agent="codex", hinted_reps=0)
    batch._adapter = codex.CodexAdapter()
    row = batch.rows[0]
    trial = Path(batch.source(row)["trial_dir"])
    rows, _ = codex_trial(trial, monkeypatch)
    rows[2]["payload"]["content"][0]["text"] = "first\u2028second\u2029third"
    raw = write_lines(batch.adapter.find_session(trial), rows)
    saved = batch.stage_bundles(row, 10)["bundles"]["control"]
    assert (Path(saved["dir"])/"session_trunc.jsonl").read_bytes() == b'\r\n'.join(raw.split(b'\r\n')[:10]) + b'\r\n'


@pytest.mark.parametrize("reason", ["stop", "error", "aborted", "missing"])
def test_pi_fake_completion(tmp_path, reason):
    batch = make_batch(tmp_path, agent="pi")
    t = tmp_path/"trial"
    (t/"agent").mkdir(parents=True)
    (t/"result.json").write_text(json.dumps(dict(finished_at="now", exception_info=None)))
    if reason != "missing":
        write_lines(t/"agent/pi.txt", [dict(type="message_end", message=dict(role="assistant", stopReason=r))
                                      for r in ("stop", reason)])
    assert batch._fake_completed(t) == (reason == "stop")


def test_pi_launch_checks_invocation_output(tmp_path):
    pi_trial(tmp_path, pi_rows())
    adapter = pi.PiAdapter()
    b, meta = bundle.build(adapter=adapter, trial_dir=tmp_path, atif_step=2, out=tmp_path/"out", agent_timeout=60)
    agent = SimpleNamespace(model_name="fixture", _OUTPUT_FILENAME="pi.txt", environment_logs_dir=Path("/logs/agent"),
                            exec_as_root=AsyncMock(), exec_as_agent=AsyncMock(), _upload_config_text=AsyncMock(),
                            _upload_session_file=AsyncMock(), _verify_seeded=AsyncMock(), adapter=adapter,
                            _resume_base_url=lambda _: "http://localhost", _resume_logs=lambda: tmp_path,
                            _bundle=lambda: (b, meta))
    for run in (pi.PiBaseline.run, NativeResumeMixin.run):
        agent.exec_as_agent.reset_mock()
        agent.exec_as_root.reset_mock()
        asyncio.run(run(agent, "task", None, None))
        commands = [c.kwargs["command"] for mock in (agent.exec_as_agent, agent.exec_as_root)
                    for c in mock.await_args_list]
        assert any("--check-output" in c and "/logs/agent/pi.txt" in c for c in commands)


@pytest.mark.parametrize("name", ["pi", "codex"])
def test_source_refuses_named_user_before_guest(tmp_path, monkeypatch, name):
    task = tmp_path/"task.toml"
    task.write_text('[agent]\nuser="worker"')
    (tmp_path/"plan.json").write_text(json.dumps({"actions": [{"tool": "Write"}]}))
    (tmp_path/"bundle.json").write_text(json.dumps(dict(run_id="r", resume_atif_step=2, agent=name)))
    guest = Mock(side_effect=AssertionError("guest access"))
    monkeypatch.setattr(source, "_sandbox_cls", guest)
    with pytest.raises((ValueError, RuntimeError), match="worker"):
        source.prepare_source(adapter=SimpleNamespace(name=name), task_dir=tmp_path, bundle=tmp_path,
                              template="t", tag="z", state_dir=tmp_path/"state")
    guest.assert_not_called()


def test_replayer_refuses_before_first_write(tmp_path, monkeypatch):
    import sys
    from agent_snapshot_resume.replay import replay_inside
    rows = pi_rows()
    rows[0]["cwd"] = str(tmp_path)
    # An ordinary, confirmed write precedes the unconfirmed write in the plan.
    rows[4].update(parentId="o", message=dict(rows[2]["message"], content=[
        dict(type="toolCall", name="write", id="second", arguments=dict(path="unconfirmed", content="x"))]))
    rows[5]["parentId"] = "old"
    record = tmp_path/"plan.json"
    ir = pi_trial(tmp_path, rows)
    record.write_text(json.dumps(plan(ir, 3)))
    monkeypatch.setattr(sys, "argv", ["replay", str(record), str(tmp_path/"report")])
    with pytest.raises(RuntimeError, match="toolResult"):
        replay_inside.main()
    assert not (tmp_path/"out").exists() and not (tmp_path/"unconfirmed").exists()


def test_codex_separate_responses(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    rows[3]["payload"]["arguments"] = json.dumps(dict(cmd="echo a > first"))
    rows[4]["payload"]["output"] = "Chunk ID: 1\nProcess exited with code 0\nOutput:\ndone"
    second = dict(rows[3], payload=dict(rows[3]["payload"], call_id="second"))
    end = dict(rows[-1], payload=dict(rows[-1]["payload"], response_id="second"))
    adapter = codex.CodexAdapter()
    write_lines(adapter.find_session(tmp_path),
                rows[:4] + [rows[-1], rows[4], second, end, dict(rows[4], payload=dict(rows[4]["payload"], call_id="second")), rows[-2]])
    ir = adapter.load_trace(tmp_path, 60)
    assert len(ir["steps"]) == 4
    p = plan(ir, 3)
    assert len(p["actions"]) == 2 and not p.get("replay_refusals")


def test_pi_failed_shell_output_is_not_validation(tmp_path):
    rows = pi_rows()
    rows[2]["message"]["content"][0].update(name="bash", arguments=dict(command="touch created && false"))
    rows[3]["message"].update(isError=True, content=[dict(type="text", text=
        'Validation failed for tool "bash":\nexample output\n\nCommand exited with code 1')])
    action, = plan(pi_trial(tmp_path, rows), 2)["actions"]
    assert action["tool"] == "Bash" and action["is_error"]


def test_pi_final_event_incomplete(tmp_path):
    from agent_snapshot_resume.proxy.pi_receipts import check_output
    path = tmp_path/"pi.txt"
    stop = dict(type="message_end", message=dict(role="assistant", stopReason="stop"))
    write_lines(path, [stop, dict(type="message_start", message=dict(role="assistant"))])
    with pytest.raises(ValueError):
        check_output(path)
    write_lines(path, [stop, dict(type="agent_end", messages=[dict(role="assistant", stopReason="error")])])
    with pytest.raises(ValueError):
        check_output(path)
