"""Chained round-N batch units: bundle build and snapshot preparation use the unit's chain spec."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agent_snapshot_resume.batch import Batch  # noqa: E402


def make_batch(tmp_path, policy="lenient"):
    b = object.__new__(Batch)
    b.name, b.work, b.tasks_dir, b.state_dir = "r2", tmp_path, tmp_path / "tasks", tmp_path / "state"
    b.batch_dir = tmp_path / "batch" / "r2"  # unit_dir() appends units/<task>
    b.jobs_dir, b.runs_dir = tmp_path / "jobs", tmp_path / "runs"
    b.state_dir.mkdir(parents=True, exist_ok=True)
    b.replay_policy, b.log, b._adapter = policy, print, None
    b.agent_name = "claude-code"
    b.manifest_path = tmp_path / "manifest.csv"
    b.hinted_reps, b.control_reps, b.midrun_every = 2, 1, 1200.0
    b.retry_failed, b.prep_concurrency, b.launch_concurrency = "off", 1, 1
    b.cut_window, b.max_cut_candidates, b.verifier, b.cost_alarm_usd = 200, 16, True, 100.0
    b.agent_env, b.passthrough_args, b.model, b.upstream_url = {}, None, None, None
    return b


def chained_unit(tmp_path, task="t", cut=900):
    """A unit dir with chain.json and a row whose hint file exists."""
    ud = tmp_path / "batch" / "r2" / "units" / task
    ud.mkdir(parents=True, exist_ok=True)
    trial = tmp_path / "jobs" / "r1-t-s7-hinteda" / "t__abc"
    (trial / "agent" / "sessions" / "projects").mkdir(parents=True, exist_ok=True)
    (trial / "agent" / "sessions" / "projects" / "s.jsonl").write_text("{}\n")
    (ud / "chain.json").write_text(json.dumps({
        "trial_dir": str(trial),
        "base_snapshot": {"snapshot_id": "snap-1", "session_line": 120},
        "prior_hints": ["first hint text"],
    }))
    hint = tmp_path / f"{task}-hint2.txt"
    hint.write_text("second hint text\n")
    return ud, trial, {"task": task, "cut_line": cut, "hint_path": str(hint), "group": "opus_never"}


def test_stage_bundles_chains_only_the_hinted_bundle(tmp_path, monkeypatch):
    from agent_snapshot_resume import bundle as bundle_mod

    b = make_batch(tmp_path)
    ud, trial, row = chained_unit(tmp_path)
    b.manifest_path = tmp_path / "manifest.csv"
    calls = []

    def fake_build(**kw):
        calls.append(kw)
        d = Path(kw["out"]) / kw["run_id"]
        d.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": kw["run_id"], "cut_line": kw["cut_line"], "orig_request_at_k": {"index": 1},
                "n_replay_actions": 3, "base_snapshot": kw.get("base_snapshot")}
        (d / "bundle.json").write_text(json.dumps(meta))
        (d / "plan.json").write_text("{}")
        return d, meta

    monkeypatch.setattr(bundle_mod, "build", fake_build)
    out = b.stage_bundles(row, row["cut_line"])
    assert [c["run_id"] for c in calls] == [f"t-s900-hint", f"t-s900-control"]
    hinted, control = calls
    assert Path(hinted["trial_dir"]) == trial.resolve()
    assert hinted["prior_hints"] == ["first hint text"]
    assert hinted["base_snapshot"]["snapshot_id"] == "snap-1"
    # the control bundle keeps the carried hint in the session but injects nothing new
    assert control["prior_hints"] is None
    assert control["base_snapshot"]["snapshot_id"] == "snap-1"
    assert control["mode"] == "same"
    assert out["inputs"]["chain"]["base_snapshot"] == "snap-1"


def test_stage_snapshot_starts_from_the_chained_base_snapshot(tmp_path, monkeypatch):
    from agent_snapshot_resume.sandbox import source

    b = make_batch(tmp_path)
    ud, trial, row = chained_unit(tmp_path)
    bdir = ud / "bundles" / "t-s900-hint"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "bundle.json").write_text(json.dumps({"base_snapshot": {"snapshot_id": "snap-1"}}))
    bundled = {"bundles": {"hinted": {"dir": str(bdir)}, "control": {"dir": str(bdir)}}}
    seen = {}

    def fake_prepare(**kw):
        seen.update(kw)
        return {"snapshot_id": "snap-2", "tag": "t-s900", "divergence_report": {"n_divergent": 0}}

    monkeypatch.setattr(source, "prepare_source", fake_prepare)
    b.stage_snapshot(row, row["cut_line"], bundled, "tmpl")
    assert seen["base_snapshot"] == "snap-1"
    assert seen["replay_policy"] == "lenient"


def test_chain_spec_must_name_a_trial_and_a_snapshot(tmp_path, monkeypatch):
    b = make_batch(tmp_path)
    ud, trial, row = chained_unit(tmp_path)
    (ud / "chain.json").write_text(json.dumps({"trial_dir": str(trial)}))
    with pytest.raises(RuntimeError, match="needs trial_dir and base_snapshot"):
        b.stage_bundles(row, row["cut_line"])
