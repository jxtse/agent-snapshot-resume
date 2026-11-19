"""Fourth native review. Fixtures stay local and exercise replay gates and bundle reuse."""
import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from agent_snapshot_resume import bundle, screen
from agent_snapshot_resume.agents.codex import CodexAdapter, CodexSnapshotResume
from agent_snapshot_resume.agents.pi import PiAdapter, tool_action
from agent_snapshot_resume.proxy import native_wire
from agent_snapshot_resume.replay.gate import validate_replay
from agent_snapshot_resume.sandbox.source import native_user_refusal
from agent_snapshot_resume.trace.ir import replay_plan
from test_batch import make_batch
from test_native_adapters import codex_trial, pi_rows, write_lines
from test_review_native import pi_trial, replay_actions
from test_review_round3 import FULL


def refused(plan, reason):
    assert screen.screen_plan(plan)["verdict"] == "block"
    for policy in ("strict", "lenient"):
        with pytest.raises(RuntimeError, match=reason):
            validate_replay(plan, {"results": []}, 0, policy=policy)


@pytest.mark.parametrize("second,blocked", [("printf b > file", True), ("cat file", False)])
def test_pi_parallel_writes(tmp_path, second, blocked):
    rows = pi_rows()
    calls = [dict(type="toolCall", name="bash", id=cid, arguments=dict(command=cmd))
             for cid, cmd in [("call", "printf a > file"), ("next", second)]]
    rows[2]["message"]["content"] = calls
    rows[4].update(parentId="o", message=dict(rows[3]["message"], toolCallId="next"))
    rows[5]["parentId"] = rows[4]["id"]
    ir = pi_trial(tmp_path, rows)
    assert not bundle.plan_with_observations(ir, 1).get("replay_refusals")
    plan = bundle.plan_with_observations(ir, 2)
    if blocked:
        refused(plan, "parallel")
    else:
        assert not plan.get("replay_refusals")


@pytest.mark.parametrize("cmd", ['cp "$PI_SESSION_FILE" checkpoint', 'cat "${PI_SESSION_FILE}" > checkpoint',
                                 "cat /logs/agent/pi/sessions/session.jsonl > checkpoint"])
def test_pi_session_file_contents(tmp_path, cmd):
    rows = pi_rows()
    rows[2]["message"]["content"][0].update(name="bash", arguments={"command": cmd})
    ir = pi_trial(tmp_path, rows)
    assert not bundle.plan_with_observations(ir, 1).get("replay_refusals")
    refused(bundle.plan_with_observations(ir, 2), "PI_SESSION_FILE")


REJECT = ('`justification` requires an explicit `sandbox_permissions`; use `sandbox_permissions: '
          '"require_escalated"` for unsandboxed execution, or omit `justification`.')
EXIT = "Chunk ID: 1\nProcess exited with code 0\nOutput:\n"


@pytest.mark.parametrize("response", [REJECT, "unknown tool response", None, "late", EXIT+REJECT])
def test_codex_confirm_execution(tmp_path, monkeypatch, response):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    rows[3]["payload"]["arguments"] = json.dumps(dict(cmd="touch output", justification="fixture"))
    rows[4]["payload"]["output"] = EXIT if response == "late" else response
    after = rows[:4] + ([] if response in (None, "late") else [rows[4]]) + [rows[5], rows[-2]]
    if response == "late":
        after.append(rows[4])
    adapter = CodexAdapter()
    write_lines(adapter.find_session(tmp_path), after)
    ir = adapter.load_trace(tmp_path, 60)
    assert not bundle.plan_with_observations(ir, 1)["actions"]
    plan = bundle.plan_with_observations(ir, 2)
    if response == REJECT:
        assert not plan["actions"] and not plan.get("replay_refusals")
        assert not replay_plan(ir, 2, include_read_only=True)["actions"]
    elif response == EXIT+REJECT:
        assert len(plan["actions"]) == 1 and not plan.get("replay_refusals")
    else:
        refused(plan, "unconfirmed")


@pytest.mark.parametrize("chars,k,blocked", [("", 2, True), ("x", 3, True), ("", 3, False)])
def test_codex_poll_before_and_after_completion(tmp_path, monkeypatch, chars, k, blocked):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    rows[6]["payload"]["arguments"] = json.dumps(dict(session_id=99, chars=chars))
    adapter = CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows)
    plan = bundle.plan_with_observations(adapter.load_trace(tmp_path, 60), k)
    if blocked:
        refused(plan, "process|poll")
        assert plan["actions"][0]["side_effect"] == "exec_background"
    else:
        action, = plan["actions"]
        assert action["side_effect"] == "exec" and not plan.get("replay_refusals")


@pytest.mark.parametrize("encoding", ["array", "object", "json-array", "json-object"])
def test_pi_edit_mixed_arguments(tmp_path, monkeypatch, encoding):
    first = dict(oldText="first", newText="one")
    edits = [first] if encoding.endswith("array") else first
    if encoding.startswith("json"):
        edits = json.dumps(edits)
    path = tmp_path/"file"
    path.write_text("first and second")
    name, args, effect = tool_action("edit", dict(path="file", edits=edits, oldText="second", newText="two"),
                                     str(tmp_path))
    action = dict(tool=name, args=args, side_effect=effect, call_id="edit")
    result, = replay_actions(tmp_path, [action], monkeypatch)
    assert result["fidelity"] == "exact" and path.read_text() == "one and two"


def test_bundle_kind_reenabled(tmp_path):
    batch = make_batch(tmp_path, agent="pi", control_reps=1, hinted_reps=1)
    batch._adapter = PiAdapter()
    row = batch.rows[0]
    trial = Path(batch.source(row)["trial_dir"])
    pi_trial(trial, pi_rows())
    write_lines(trial/"agent/model-proxy/native-reference.jsonl", [dict(
        session_id="session", entry_id="a", content=native_wire.body_reference("chat", {"messages": []}))])
    first = batch.stage_bundles(row, 5)["bundles"]
    batch.control_reps = 0
    (batch.task_dir(row)/"task.toml").write_text("[agent]\ntimeout_sec=900")
    assert set(batch.stage_bundles(row, 5)["bundles"]) == {"hinted"}
    batch.control_reps = 1
    again = batch.stage_bundles(row, 5)["bundles"]
    assert again["hinted"]["agent_budget_s"] != first["hinted"]["agent_budget_s"]
    assert again["control"]["agent_budget_s"] == again["hinted"]["agent_budget_s"]


def test_codex_provisional_usage_replaced(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    session = CodexAdapter().find_session(tmp_path)
    (tmp_path/"session_trunc.jsonl").write_bytes(session.read_bytes())

    def usage(n):
        return dict(input_tokens=n, output_tokens=2, cached_input_tokens=0)

    def count(n):
        return dict(rows[5], payload=dict(type="token_count", info=dict(last_token_usage=usage(n))))

    receipt = dict(rows[-1], payload=dict(response_id="new", usage=usage(10)))
    write_lines(session, rows + [rows[-2], count(9), count(9), receipt, receipt, rows[-2], count(10)])
    agent = CodexSnapshotResume(logs_dir=tmp_path/"agent", model_name="some-model")
    agent._bundle = lambda: (tmp_path, dict(session_relpath=str(session.relative_to(tmp_path))))
    monkeypatch.setattr(agent, "_compute_cost_from_pricing", lambda **kw: kw["prompt_tokens"] * .01)
    ctx = AgentContext()
    agent.populate_context_post_run(ctx)
    assert (ctx.n_input_tokens, ctx.n_output_tokens, ctx.cost_usd) == (20, 4, .2)
    metrics = json.loads((tmp_path/"agent/trajectory.json").read_bytes())["final_metrics"]
    assert metrics["extra"]["resumed_calls"] == 2


def manual_plan(tmp_path, command, output, rc, **fields):
    rows = pi_rows()
    rows[0]["cwd"] = str(tmp_path)
    rows[2]["message"]["content"] = [dict(type="text", text="first")]
    rows[3]["message"] = dict(role="bashExecution", command=command, output=output, exitCode=rc, **fields)
    return bundle.plan_with_observations(pi_trial(tmp_path, rows), 3)


def test_pi_manual_full_output(tmp_path):
    plan = manual_plan(tmp_path, "cat large", "tail", 0, fullOutputPath="/tmp/pi-bash-aabb.log", truncated=True)
    refused(plan, "full output")


@pytest.mark.parametrize("code,output", [(1, ""), (7, "out\nerr\n"), (2, "Exit code text")])
def test_pi_manual_exit_code(tmp_path, monkeypatch, code, output):
    import shlex
    cmd = "printf %s " + shlex.quote(output) + f" >&2; exit {code}"
    plan = manual_plan(tmp_path, cmd, output, code)
    result, = replay_actions(tmp_path, plan["actions"], monkeypatch)
    assert result["fidelity"] in ("exact", "normalized")
    for policy in ("strict", "lenient"):
        validate_replay(plan, {"results": [result]}, 0, policy=policy)
        with pytest.raises(RuntimeError, match="exit_code"):
            validate_replay(plan, {"results": [dict(result, rc=code+1)]}, 0, policy=policy)


@pytest.mark.parametrize("user", ["0", '"0"', '"root"', "1000"])
def test_root_user_forms(tmp_path, user):
    (tmp_path/"task.toml").write_text(f"[agent]\nuser={user}")
    for name in ("pi", "codex"):
        issue = native_user_refusal(name, tmp_path, {"actions": [1]}, "strict")
        assert bool(issue) == (user == "1000")


def test_codex_invalid_json_rejection(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"].update(FULL)
    rows[3]["payload"]["arguments"] = "{"
    rows[4]["payload"]["output"] = "failed to parse function arguments: EOF while parsing an object"
    adapter = CodexAdapter()
    write_lines(adapter.find_session(tmp_path), rows[:6] + [rows[-2]])
    ir = adapter.load_trace(tmp_path, 60)
    plan = bundle.plan_with_observations(ir, 2)
    assert not plan["actions"] and not plan.get("replay_refusals")
