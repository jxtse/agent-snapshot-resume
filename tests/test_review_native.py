"""Regressions for the native harness review, exercised without a provider."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_snapshot_resume import bundle, preflight
from agent_snapshot_resume.agents.codex import CodexAdapter, CodexSnapshotResume
from agent_snapshot_resume.agents.native import NativeResumeMixin
from agent_snapshot_resume.agents.pi import PiAdapter, PiBaseline, tool_action
from agent_snapshot_resume.proxy import native_wire
from agent_snapshot_resume.replay import replay_inside as replay
from agent_snapshot_resume.replay.gate import validate_replay
from test_native_adapters import codex_trial, pi_rows, write_lines


def pi_trial(root, rows):
    write_lines(root / "agent/pi/sessions/session.jsonl", rows)
    (root / "agent/pi/version.txt").write_text("1.4.0")
    return PiAdapter().load_trace(root, 60)


def replay_actions(root, actions, monkeypatch):
    plan = root / "plan.json"
    report = root / "report.json"
    plan.write_text(json.dumps(dict(initial_cwd=str(root), actions=actions)))
    monkeypatch.setattr(sys, "argv", ["replay", str(plan), str(report)])
    replay.main()
    return json.loads(report.read_text())["results"]


def bash_plan(root, command, observation, error=False, details=None, **args):
    rows = pi_rows()
    rows[2]["message"]["content"] = [dict(type="toolCall", name="bash", id="call",
                                          arguments=dict(command=command, **args))]
    rows[3]["message"].update(isError=error, details=details, content=[dict(type="text", text=observation)])
    rows[0]["cwd"] = str(root)
    return bundle.plan_with_observations(pi_trial(root, rows), 2)["actions"]


def test_truncated_read_only_restores_bytes(tmp_path, monkeypatch):
    path = "/tmp/pi-bash-0123456789abcdef.log"
    data = b"\xff\n" + b"x\n" * 2000
    (tmp_path / "long").write_bytes(data)
    obs = "\n".join(["x"] * 2000) + "\n\n[Showing lines 2-2001 of 2001. Full output: " + path + "]"
    actions = bash_plan(tmp_path, "cat long", obs, details={"fullOutputPath": path})
    assert len(actions) == 1
    # Redirect only the guest log directory during this unit test.
    monkeypatch.setattr(replay, "PI_LOG_DIR", str(tmp_path))
    result, = replay_actions(tmp_path, actions, monkeypatch)
    assert result["fidelity"] == "exact"
    assert (tmp_path / Path(path).name).read_bytes() == data
    assert result["pi_output_sha256"] == hashlib.sha256(data).hexdigest()
    for policy in ("strict", "lenient"):
        with pytest.raises(RuntimeError, match="pi_output_missing"):
            validate_replay({"actions": actions}, {"results": [dict(result, pi_output_sha256=None)]},
                            0, policy=policy)
    from agent_snapshot_resume.sandbox.source import _collect_gate_evidence
    rec = {}
    files = Mock()
    files.read.return_value = data
    report = json.loads((tmp_path / "report.json").read_bytes())
    _collect_gate_evidence(SimpleNamespace(files=files), rec, report, Mock())
    assert rec["auxiliary_hashes"] == {path: result["pi_output_sha256"]}
    files.read.return_value = b"changed"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _collect_gate_evidence(SimpleNamespace(files=files), rec, report, Mock())


@pytest.mark.parametrize("receipt", [True, False])
def test_seeded_codex_usage(tmp_path, monkeypatch, receipt):
    from harbor.models.agent.context import AgentContext
    rows, _ = codex_trial(tmp_path, monkeypatch)
    session = CodexAdapter().find_session(tmp_path)
    seed = tmp_path / "session_trunc.jsonl"
    seed.write_bytes(session.read_bytes())
    def usage(n):
        return dict(input_tokens=n, output_tokens=n, cached_input_tokens=0)
    def event(kind, **p):
        return dict(timestamp="2026-01-01T00:00:03Z", type=kind, payload=p)
    write_lines(session, rows + [
        event("response_item", type="message", role="assistant", content=[dict(type="output_text", text="resumed")]),
        *([event("token_usage_record", response_id="next", usage=usage(1))] if receipt else []),
        event("event_msg", type="token_count", info=dict(total_token_usage=usage(72), last_token_usage=usage(1)))])
    agent = CodexSnapshotResume(logs_dir=tmp_path/"agent", model_name="some-model")
    agent._bundle = lambda: (tmp_path, dict(session_relpath=str(session.relative_to(tmp_path))))
    monkeypatch.setattr(agent, "_compute_cost_from_pricing", lambda **kw: (kw["prompt_tokens"] or 0) * .01)
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.n_input_tokens == context.n_output_tokens == 1
    assert context.cost_usd == .01
    trajectory = json.loads((tmp_path/"agent/trajectory.json").read_text())
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 1
    assert len(trajectory["steps"]) > 1 and trajectory["steps"][-1]["message"].endswith("resumed")


def test_codex_cut_cwd(tmp_path, monkeypatch):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    rows[1]["payload"]["cwd"] = "/changed"
    write_lines(CodexAdapter().find_session(tmp_path), rows)
    b, meta = bundle.build(adapter=CodexAdapter(), trial_dir=tmp_path, atif_step=4,
                           agent_timeout=60, out=tmp_path/"out")
    assert meta["cwd"] == "/changed"
    assert json.loads((b/"plan.json").read_text())["initial_cwd"] == "/work"
    assert json.loads((b/"proxy_spec.json").read_text())["resume_artifacts"]["cwd"] == "/changed"
    CodexAdapter().sanity_check_bundle(b, meta)
    with pytest.raises(ValueError, match="working directory"):
        CodexAdapter().sanity_check_bundle(b, dict(meta, cwd="/work"))


def test_pi_path_resolution():
    def path(value, **kwargs):
        return tool_action("write", dict(path=value, content="x"), "/task", **kwargs)[1]["file_path"]
    assert path("@name") == "/task/name"
    assert path("@file://localhost/a%20b") == "/a b"
    assert path("~/test", home="/users/me") == "/users/me/test"
    with pytest.raises(ValueError):
        path("~/test")
    with pytest.raises(ValueError):
        path("file://remote/test")


def test_pi_reverse_alignment(tmp_path):
    rows = pi_rows()
    rows[-1]["parentId"] = rows[-2]["id"]
    rows[-1]["timestamp"] = rows[-2]["timestamp"]
    (tmp_path/"agent").mkdir()
    (tmp_path/"agent/trajectory.json").write_text(json.dumps(dict(steps=[
        dict(source="agent", timestamp=rows[-1]["timestamp"], step_id=90)])))
    ir = pi_trial(tmp_path, rows)
    assert [s["atif_step_id"] for s in ir["steps"][-2:]] == [None, None]
    with pytest.raises(SystemExit, match="no aligned ATIF step"):
        bundle.build(adapter=PiAdapter(), trial_dir=tmp_path, atif_step=None,
                     cut_line=ir["steps"][-1]["line_range"][0], agent_timeout=60, out=tmp_path/"out")


@pytest.mark.parametrize("key", ["session", "session_id", "thread_id", "response_id", "entry_id", "model"])
def test_codex_conflicting_identity(tmp_path, monkeypatch, key):
    rows, _ = codex_trial(tmp_path, monkeypatch)
    adapter = CodexAdapter()
    ir = adapter.load_trace(tmp_path, 60)
    log = tmp_path/"agent/model-proxy/requests-1.jsonl"
    row = json.loads(log.read_text())
    prefix = [json.dumps(r) for r in rows[:10]]
    write_lines(log, [dict(row, **{key: "unrelated"})])
    assert adapter.native_reference(tmp_path, ir, 3, prefix)[0] is None
    write_lines(log, [dict(row, **{key: "unrelated"}), row])
    assert adapter.native_reference(tmp_path, ir, 3, prefix)[0]["index"] == 1


def test_pi_request_start_budget(tmp_path):
    rows = pi_rows()
    rows[-1]["message"]["timestamp"] = 1767225603100
    last = pi_trial(tmp_path, rows)["steps"][-1]
    assert last["timestamp"] == rows[-1]["timestamp"]
    assert last["elapsed_s"] == 3.1
    assert last["remaining_s"] == 56.9


@pytest.mark.parametrize("multi", [False, True])
def test_pi_edit_bytes(tmp_path, monkeypatch, multi):
    path = tmp_path / "file"
    path.write_bytes(b"\xef\xbb\xbfa\r\nb\r\nc\r\n")
    edits = [dict(oldText="a\nb", newText="A\nB")]
    if multi:
        edits.append(dict(oldText="c", newText="C"))
    name, args, effect = tool_action("edit", dict(path="file", edits=edits), str(tmp_path))
    res, = replay_actions(tmp_path, [dict(call_id="c", tool=name, args=args, side_effect=effect)], monkeypatch)
    assert res["fidelity"] == "exact"
    assert path.read_bytes() == b"\xef\xbb\xbfA\r\nB\r\n" + (b"C" if multi else b"c") + b"\r\n"


@pytest.mark.parametrize("wire,key", [("responses", "input"), ("chat", "messages")])
def test_native_preflight(tmp_path, wire, key):
    body = dict(model="fixture", **{key: [native_wire.user_item(wire, "original")]})
    ref, obs = tmp_path/"ref", tmp_path/"obs"
    ref.write_text(json.dumps(body))
    obs.write_text(ref.read_text())
    assert preflight.request_matches(ref, obs, None)
    body[key].append(native_wire.user_item(wire, "hint"))
    obs.write_text(json.dumps(body))
    hint = dict(message=1, block=0, text="hint")
    assert preflight.request_matches(ref, obs, hint)
    assert not preflight.request_matches(ref, obs, dict(hint, message=0))
    body["model"] = "other"
    obs.write_text(json.dumps(body))
    assert not preflight.request_matches(ref, obs, hint)


def test_interrupted_pi_receipt(tmp_path):
    rows = pi_rows()
    rows[2]["message"]["responseId"] = "completed"
    rows[-1]["message"]["stopReason"] = "aborted"
    ir = pi_trial(tmp_path, rows)
    ref = dict(wire="chat", complete=True, raw_sha256="a"*64)
    receipt = dict(path="/v1/chat/completions", status=200, response_id="completed", native_reference=ref)
    log = tmp_path/"agent/model-proxy/requests-1.jsonl"
    write_lines(log, [receipt, dict(receipt, response_id="orphan")])
    receipt_at_cut, _ = PiAdapter().native_reference(tmp_path, ir, 1, [])
    assert receipt_at_cut and receipt_at_cut["content"] == ref
    write_lines(log, [receipt, receipt])
    assert PiAdapter().native_reference(tmp_path, ir, 1, [])[0] is None


def test_large_native_upload(tmp_path):
    from agent_snapshot_resume.agents.claude_code import UPLOAD_PART_BYTES
    seed = tmp_path/"session_trunc.jsonl"
    seed.write_bytes(b"x"*(UPLOAD_PART_BYTES+9))
    agent = NativeResumeMixin()
    agent.adapter = Mock()
    import shutil
    import subprocess
    guest = tmp_path/"guest"
    session = guest/"session.jsonl"
    agent.adapter.resume_setup.return_value = ({}, str(session), ["pi"], str(guest/"cfg"), "{}")
    agent._bundle = lambda: (tmp_path, dict(run_id="r", aux_counts={}, cwd="/", agent_budget_s=60))
    agent._resume_logs = lambda: tmp_path
    async def execute(_, *, command, **kwargs):
        if "timeout -s TERM" in command:  # seed verification has finished
            return
        return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    agent.exec_as_agent = agent.exec_as_root = execute
    agent.environment_logs_dir = tmp_path
    agent._OUTPUT_FILENAME = "stdout"
    sizes = []
    async def upload(src, dst):
        sizes.append(src.stat().st_size)
        shutil.copyfile(src, dst)
    asyncio.run(agent.run("", SimpleNamespace(upload_file=upload, default_user=None), None))
    assert max(sizes) <= UPLOAD_PART_BYTES
    assert sum(sizes) == seed.stat().st_size + 2
    hashes = json.loads((tmp_path/"session_aux_hashes.json").read_bytes())
    assert hashes["actual"] == hashes["expected"]
    assert hashes["actual"][str(session)] == hashlib.sha256(seed.read_bytes()).hexdigest()


def test_pi_timeout_prevents_late_write(tmp_path, monkeypatch):
    actions = bash_plan(tmp_path, "printf begun; sleep 2; touch late", "begun\n\nCommand timed out after 1 seconds",
                        error=True, timeout=1)
    result, = replay_actions(tmp_path, actions, monkeypatch)
    assert not (tmp_path/"late").exists()
    assert result["fidelity"] == "exact" and result["termination_reproduced"]
    validate_replay({"actions": actions}, {"results": [result]}, 0)
    for policy in ("strict", "lenient"):
        with pytest.raises(RuntimeError, match="pi_timeout_changed"):
            validate_replay({"actions": actions}, {"results": [dict(result, termination_reproduced=False)]},
                            0, policy=policy)


def test_pi_namespaced_model():
    agent = SimpleNamespace(model_name="asr-reasoning/org/model", exec_as_root=AsyncMock(),
                            exec_as_agent=AsyncMock(), _upload_config_text=AsyncMock(), _OUTPUT_FILENAME="output")
    asyncio.run(PiBaseline.run(agent, "task", None, None))
    config = json.loads(agent._upload_config_text.await_args.kwargs["content"])
    assert config["providers"]["asr-reasoning"]["models"][0]["id"] == "org/model"
    assert "--model org/model" in agent.exec_as_agent.await_args.kwargs["command"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("outcome", ["exception", "missing", "unfinished"])
def test_fake_requires_terminal_success(tmp_path, monkeypatch, cached, outcome):
    from test_batch import make_batch, fake_build, fake_launch
    from agent_snapshot_resume import launch
    from agent_snapshot_resume.bundle import snapshot_cut_identity
    batch = make_batch(tmp_path)
    row = batch.rows[0]
    unit = batch.unit_dir(row["task"])
    bdir, _ = fake_build(adapter=None, trial_dir=None, atif_step=None, cut_line=10, out=unit, task_dir=None,
                         mode="same", hint=None, run_id="bundle")
    bundled = dict(bundles=dict(control=dict(dir=str(bdir))))
    def run(**kw):
        rc = fake_launch([])(**kw)
        trial = batch.find_trial(kw["job_name"])
        result = trial/"result.json"
        if outcome == "missing":
            result.unlink(missing_ok=True)
        elif outcome == "unfinished":
            result.write_text(json.dumps(dict(finished_at=None, exception_info=None)))
        else:
            result.write_text(json.dumps(dict(finished_at="2026-01-01T00:00:05Z",
                                              exception_info=dict(exception_type="AgentTimeoutError"))))
        return rc
    if cached:
        job = batch.job(row, "fake-control")
        run(jobs_dir=batch.jobs_dir, runs_dir=batch.runs_dir, job_name=job)
        trial = batch.find_trial(job)
        first = next(trial.rglob("first_request_rewritten.json"))
        ident = trial/"agent/resume/bundle_identity.json"
        ident.parent.mkdir()
        ident.write_text(json.dumps(dict(cut=snapshot_cut_identity(bdir))))
        (unit/"fake.json").write_text(json.dumps(dict(runs=dict(control=dict(admitted=True, first_request=str(first))))))
        monkeypatch.setattr(launch, "frozen_first_request_digest", lambda *_: "ok")
    monkeypatch.setattr(launch, "launch_one", run)
    with pytest.raises(RuntimeError, match="fake.*(terminal|execution|result)"):
        batch.stage_fake(row, 10, bundled, tmp_path/"snap")



def test_batch_rebuilds_old_native_plan(tmp_path, monkeypatch):
    from test_batch import make_batch, fake_build
    batch = make_batch(tmp_path, agent="pi")
    batch.adapter.wire = "chat"
    monkeypatch.setattr(batch, "source_session_sha", lambda _: "a"*64)
    build = Mock(side_effect=fake_build)
    monkeypatch.setattr(bundle, "build", build)
    row = batch.rows[0]
    batch.stage_bundles(row, 10)
    stored = batch.unit_dir(row["task"])/"bundles.json"
    old = json.loads(stored.read_text())
    old["inputs"].pop("native_bundle_revision", None)
    stored.write_text(json.dumps(old))
    build.reset_mock()
    batch.stage_bundles(row, 10)
    assert build.called
    build.reset_mock()
    batch.stage_bundles(row, 10)
    build.assert_not_called()
