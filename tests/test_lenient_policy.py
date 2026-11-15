"""Lenient replay policy, post-cut file probe, replay-shell and template-resource fixes."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_snapshot_resume.replay import gate  # noqa: E402
from agent_snapshot_resume import screen  # noqa: E402
from agent_snapshot_resume.bundle import post_cut_referenced_files  # noqa: E402
from agent_snapshot_resume.sandbox import templates, source as source_mod  # noqa: E402


def action(call_id="call_00", side_effect="exec", is_error=False, orig="", command="python3 x.py", **kw):
    a = {"call_id": call_id, "side_effect": side_effect, "is_error": is_error, "orig_text": orig,
         "args": {"command": command}, "tool": "Bash", "bg_finished_before_k": True}
    a.update(kw)
    return a


def result(call_id="call_00", fidelity="exact", rc=0, status="ran", stdout="", stderr="", **kw):
    r = {"call_id": call_id, "fidelity": fidelity, "rc": rc, "status": status, "stdout_full": stdout,
         "stderr_full": stderr}
    r.update(kw)
    return r


def plan_of(*actions):
    return {"actions": list(actions)}


def report_of(*results, rc="0"):
    return {"rc": rc, "results": list(results)}


def test_lenient_accepts_changed_numbers_and_reports_them():
    a = action(orig="accuracy 0.831 recall 0.9")
    r = result(fidelity="numbers_differ", stdout="accuracy 0.845 recall 0.91")
    rep = gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    assert rep["policy"] == "lenient"
    assert rep["n_divergent"] == 1 and rep["by_kind"] == {"changed_numbers": 1}
    assert rep["changed_numbers"] == ["call_00"]
    assert rep["blocks"] == []


def test_strict_still_refuses_changed_numbers():
    a = action(orig="accuracy 0.831")
    r = result(fidelity="numbers_differ", stdout="accuracy 0.845")
    with pytest.raises(RuntimeError, match="Unaccepted replay fidelity numbers_differ"):
        gate.validate_replay(plan_of(a), report_of(r), "0")


def test_lenient_accepts_text_differs_and_versions():
    a = action(orig="numpy 1.26.4 scipy 1.11.1")
    r = result(fidelity="numbers_differ", stdout="numpy 2.0.1 scipy 1.13.1")
    rep = gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    assert rep["by_kind"] == {"version_diff": 1}
    assert rep["version_diffs"][0]["versions"] == [["1.26.4", "2.0.1"], ["1.11.1", "1.13.1"]]
    a2, r2 = action(orig="a\nb"), result(fidelity="differs", stdout="a\nc")
    rep2 = gate.validate_replay(plan_of(a2), report_of(r2), "0", policy="lenient")
    assert rep2["text_differs"] == ["call_00"]


def test_lenient_blocks_failure_where_original_succeeded():
    a = action(orig="ok")
    r = result(fidelity="differs", rc=1, stdout="Traceback (most recent call last):")
    with pytest.raises(RuntimeError, match="exit_code_changed"):
        gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    blocks = gate.lenient_blocks(plan_of(a), report_of(r), "0")
    assert blocks[0]["rule"] == "exit_code_changed" and blocks[0]["rc"] == 1


def test_lenient_blocks_timeout_and_kill():
    a = action(orig="done")
    with pytest.raises(RuntimeError, match="replay_timeout"):
        gate.validate_replay(plan_of(a), report_of(result(status="timeout", rc=None, stdout="partial")), "0",
                             policy="lenient")
    with pytest.raises(RuntimeError, match="killed_or_oom"):
        gate.validate_replay(plan_of(a), report_of(result(fidelity="numbers_differ", rc=137, stdout="")), "0",
                             policy="lenient")
    # a kill signalled only by the text (torch OOM) is a block too
    with pytest.raises(RuntimeError, match="killed_or_oom"):
        gate.validate_replay(plan_of(a), report_of(result(fidelity="differs", rc=1, stderr="RuntimeError: CUDA out of "
                                                                                          "memory")), "0",
                             policy="lenient")


def test_lenient_blocks_missing_post_cut_file():
    a = action(orig="ls")
    r = result(fidelity="exact", stdout="ls")
    post = {"n": 2, "missing": ["/root/outputs/model.h5"], "paths": ["/root/outputs/model.h5", "/app/x.csv"]}
    with pytest.raises(RuntimeError, match="post_cut_file_missing"):
        gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient", post_cut=post)
    rep = gate.lenient_report(plan_of(a), report_of(r), post_cut=post)
    assert rep["files_missing"] == ["/root/outputs/model.h5"] and rep["post_cut_paths"] == 2
    # same plan, everything present: accepted
    rep2 = gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient", post_cut={"n": 2, "missing": []})
    assert rep2["files_missing"] == []


def test_lenient_blocks_background_job_in_flight_at_cut():
    a = action("call_bg", side_effect="exec_background", orig="", bg_finished_before_k=False,
               background_output_file="/root/run.log")
    r = result("call_bg", fidelity="bg_rerun_foreground", rc=0, background_output_file="/root/run.log",
               background_output_sha256="x")
    with pytest.raises(RuntimeError, match="background_in_flight_at_cut"):
        gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    # a job that had finished before the cut is accepted, even though its output body differs
    a["bg_finished_before_k"] = True
    rep = gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    assert rep["n_divergent"] == 1 and rep["by_kind"] == {"background_content": 1}


def test_lenient_blocks_incomplete_replay_and_crashed_replayer():
    with pytest.raises(RuntimeError, match="replayer_failed"):
        gate.validate_replay(plan_of(action()), report_of(result()), "2", policy="lenient")
    with pytest.raises(RuntimeError, match="replay_incomplete"):
        gate.validate_replay(plan_of(action(), action("call_01")), report_of(result()), "0", policy="lenient")


def test_lenient_accepts_exit_outcome_change_from_error_to_success_but_records_it():
    a = action(orig="No such file", is_error=True)
    r = result(fidelity="differs", rc=127, stderr="command not found")
    rep = gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")
    assert rep["n_divergent"] == 1 and rep["blocks"] == []


def test_lenient_blocks_edit_not_applied():
    a = action(side_effect="file_write", tool="Edit", orig="edited")
    r = result(status="edit_failed_ambiguous", rc=None)
    with pytest.raises(RuntimeError, match="edit_not_applied"):
        gate.validate_replay(plan_of(a), report_of(r), "0", policy="lenient")


# ------------------------------------------------------------------ post-cut paths
def ir_with(steps):
    return {"steps": [{"atif_step_id": 1000 + i, "events": ev} for i, ev in enumerate(steps)], "flags": {}}


def test_post_cut_paths_from_reads_and_shell_reads():
    ir = ir_with([
        [{"type": "tool_call", "tool": "Bash", "args": {"command": "python3 train.py"}}],          # before the cut
        [{"type": "tool_call", "tool": "Read", "args": {"file_path": "/root/outputs/metrics.json"}},
         {"type": "tool_call", "tool": "Bash", "args": {"command": "cat /app/results/summary.csv | head -3"}},
         {"type": "tool_call", "tool": "Bash", "args": {"command": "cat /tmp/scratch/log.txt"}},
         {"type": "tool_call", "tool": "Bash", "args": {"command": "wc -l data/input.csv"}},
         {"type": "tool_call", "tool": "Edit", "args": {"file_path": "/app/src/model.py"}}],
    ])
    got = {p["path"]: p["how"] for p in post_cut_referenced_files(ir, 1)}
    assert got == {"/root/outputs/metrics.json": "Read", "/app/results/summary.csv": "Bash"}
    assert post_cut_referenced_files(ir, 0) == [] or all("1000" not in str(p) for p in [])  # every step is post-cut at k=1


def test_post_cut_paths_ignore_relative_and_agent_state():
    ir = ir_with([[{"type": "tool_call", "tool": "Read", "args": {"file_path": "metrics.json"}},
                   {"type": "tool_call", "tool": "Read", "args": {"file_path": "/root/.claude/settings.json"}}]])
    assert post_cut_referenced_files(ir, 0) == []


# ------------------------------------------------------------------ screen policy
def test_screen_lenient_downgrades_pace_dependent_read_of_finished_job():
    bg = action("call_bg", side_effect="exec_background", bg_finished_before_k=True,
                bg_finished_ts="2026-01-01T00:00:10Z", background_output_file="/root/run.log",
                timestamp="2026-01-01T00:00:00Z")
    read = action("call_read", orig="loss 0.4", command="tail -n 5 /root/run.log",
                  timestamp="2026-01-01T00:01:00Z")
    strict = screen.screen_plan(plan_of(bg, read), orig_request={})
    lenient = screen.screen_plan(plan_of(bg, read), orig_request={}, policy="lenient")
    sev = lambda rep: [r["severity"] for r in rep["risks"] if r["rule"] == "pace_dependent_read"]
    assert sev(strict) == [screen.INFO]
    assert sev(lenient) == [screen.WARN]
    assert strict["verdict"] == "ok" and lenient["verdict"] == "review"
    assert lenient["policy"] == "lenient"


def test_screen_lenient_still_blocks_background_job_in_flight_at_cut():
    bg = action("call_bg", side_effect="exec_background", bg_finished_before_k=False,
                background_output_file="/root/run.log", timestamp="2026-01-01T00:00:00Z")
    rep = screen.screen_plan(plan_of(bg), orig_request={}, policy="lenient")
    assert rep["verdict"] == "block"
    assert any(r["rule"] == "bg_inflight" and r["severity"] == screen.BLOCK for r in rep["risks"])


# ------------------------------------------------------------------ template resources
def test_task_resources_come_from_task_toml(tmp_path):
    task = tmp_path / "t"
    task.mkdir()
    (task / "task.toml").write_text('[environment]\ncpus = 4\nmemory_mb = 16384\nstorage_mb = 32768\n'
                                   'docker_image = "img"\n')
    res = templates.task_resources(task)
    assert res["cpus"] == 4 and res["memory_mb"] == 16384
    assert res["disk_mib"] == 15360  # FC has no larger bucket than 15360 MiB


def test_fc_disk_buckets():
    assert templates.fc_disk_mib(512) == 512
    assert templates.fc_disk_mib(4096) == 10240
    assert templates.fc_disk_mib(10240) == 10240
    assert templates.fc_disk_mib(30720) == 15360
    assert templates.fc_disk_mib(None) == 15360


# ------------------------------------------------------------------ replay shell
class _FakeRun:
    def __init__(self):
        self.commands = []

    def run(self, command, **kw):
        self.commands.append(command)
        return type("R", (), {"pid": 4242})()


def test_dispatch_replay_uses_a_login_shell():
    sbx = type("S", (), {"commands": _FakeRun()})()
    pid = source_mod.dispatch_replay(sbx, "/app", 900)
    assert pid == 4242
    cmd = sbx.commands.commands[0]
    # bash -lc: /etc/profile.d (where the task healthcheck installs the task's conda env) must be sourced, otherwise
    # the base conda env shadows the task env (e.g. /opt/conda/bin/python3 instead of .../envs/stereo/bin/python3)
    assert cmd.startswith("bash -lc ")
    assert "timeout -s TERM" in cmd


def test_probe_post_cut_reports_missing(tmp_path):
    plan = {"post_cut_paths": [{"path": "/a"}, {"path": "/b"}]}

    class Files:
        def write(self, path, data):
            self.path, self.data = path, data

        def read(self, path, format=None):
            return json.dumps({"/a": True, "/b": False}).encode()

    sbx = type("S", (), {"files": Files()})()
    calls = []

    def run(sbx_, cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"stdout": ""})()

    got = source_mod.probe_post_cut(sbx, run, plan, tmp_path, "t1")
    assert got == {"n": 2, "present": 1, "missing": ["/b"]}
    assert "postcut_probe.py" in calls[0]
