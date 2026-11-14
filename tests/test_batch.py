"""Batch driver: manifest parsing, cut-window selection, stage machine, idempotent resume, launch planning, status.

Everything external (screen, bundle build, template, snapshot, Harbor launch) is faked, so this exercises the driver
itself: which cut it picks, that a prepared unit is not re-prepared or re-launched, and how status classifies runs.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_snapshot_resume import batch

TASK = "sum-task"
HINT = "Decision 1: sum the column."


def manifest(tmp: Path, rows=None) -> Path:
    rows = rows or [{"task": TASK, "group": "opus_partial", "source_trial": f"{TASK}__abc", "cut_line": 100,
                     "after_compaction": "True", "n_forks": "1", "class": "single_fork", "hint_path": "",
                     "leak_ok": "True"}]
    p = tmp / "manifest.csv"
    for r in rows:
        r["hint_path"] = r.get("hint_path") or str(tmp / "hint.txt")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp / "hint.txt").write_text(HINT)
    (tmp / TASK).mkdir(exist_ok=True)
    (tmp / TASK / "source.json").write_text(json.dumps({"trial_dir": str(tmp / "l0" / f"{TASK}__abc")}))
    (tmp / "tasks" / TASK).mkdir(parents=True, exist_ok=True)
    (tmp / "tasks" / TASK / "task.toml").write_text("[agent]\ntimeout_sec = 600\n")
    (tmp / "l0" / f"{TASK}__abc").mkdir(parents=True, exist_ok=True)
    (tmp / TASK / "fork.json").write_text(json.dumps({"forks": [{"session_line": 140}]}))
    return p


def make_batch(tmp: Path, **kw) -> batch.Batch:
    args = dict(name="r1", work_dir=tmp,
                manifest=(tmp / "manifest.csv") if (tmp / "manifest.csv").is_file() else manifest(tmp),
                tasks_dir=tmp / "tasks",
                jobs_dir=tmp / "jobs", runs_dir=tmp / "runs", state_dir=tmp / "state",
                prep_concurrency=1, launch_concurrency=1)
    args.update(kw)
    b = batch.Batch(**args)
    b._adapter = FakeAdapter()
    return b


class FakeAdapter:
    """Only what the screen stage needs from an adapter: the IR (for agent-step starts), nothing else."""
    name = "claude-code"

    def __init__(self, starts=(96, 100, 104, 108)):
        self.starts = starts

    def load_trace(self, trial_dir, **kw):
        return {"error": None, "steps": [{"source": "agent", "line_range": [s, s + 1], "atif_step_id": i}
                                         for i, s in enumerate(self.starts)]}


class Screen:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def __call__(self, adapter, trial_dir, *, cut_lines=None, **kw):
        self.calls.append(list(cut_lines))
        return [{"cut_line": c, "verdict": self.verdicts.get(c, "ok"), "counts": {}, "n_actions": 1,
                 "risks": [], "trial": TASK, "atif_step": 10, "replay_exec_s": 1.0} for c in cut_lines]


def fake_launch(launches):
    """Stand-in for launch_one: writes the run dir plus the trial evidence a fake run leaves (proxy logs)."""
    def run(**kw):
        launches.append(kw)
        d = Path(kw["runs_dir"]) / kw["job_name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text("{}")
        trial = Path(kw["jobs_dir"]) / kw["job_name"] / "sum-task__t1"
        proxy = trial / "agent" / "resume-proxy" / "run-1"
        proxy.mkdir(parents=True, exist_ok=True)
        (proxy / "first_request_rewritten.json").write_text(json.dumps({"body": {"messages": []}, "actions": []}))
        (proxy / "first_request_parity.json").write_text(json.dumps({"ok": True, "request_sha256": "x"}))
        return 0
    return run


def fake_prepare(**kw):
    rec = {"snapshot_id": "snap-1", "replay_gate_passed": True, "cut_identity": {}, "tag": kw["tag"]}
    Path(kw["state_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(kw["state_dir"]) / f"snapshot-{kw['tag']}.json").write_text(json.dumps(rec))
    return rec


def fake_build(*, adapter, trial_dir, atif_step, cut_line, out, task_dir, mode, hint, run_id, **kw):
    bdir = Path(out) / run_id
    bdir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "cut_line": cut_line, "agent_budget_s": 1000.0, "n_replay_actions": 3,
            "orig_request_at_k": {"req_bytes": 10}, "mode": mode, "hint": hint,
            # fields of the cut identity: the driver compares a cached fake run's recorded identity with the bundle
            "trial": TASK, "session_sha256": "0" * 64, "resume_atif_step": 3,
            "fingerprint_roots": ["/app"], "fingerprint_optional_roots": []}
    (bdir / "bundle.json").write_text(json.dumps(meta))
    (bdir / "plan.json").write_text(json.dumps({"actions": [], "initial_cwd": "/app"}))
    return bdir, meta


class Prep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_prep(self, b, screen, launches=None):
        launches = launches if launches is not None else []
        with mock.patch("agent_snapshot_resume.bundle.resolve_agent_timeout", lambda *a, **k: 600.0), \
                mock.patch("agent_snapshot_resume.screen.screen_trial", screen), \
                mock.patch("agent_snapshot_resume.bundle.build", fake_build), \
                mock.patch("agent_snapshot_resume.sandbox.templates.build_template",
                           lambda **kw: {"template_id": "tpl-1", "status": "ready"}), \
                mock.patch("agent_snapshot_resume.sandbox.source.prepare_source", fake_prepare), \
                mock.patch("agent_snapshot_resume.launch.launch_one", fake_launch(launches)):
            return b.run(do_launch=True)

    def test_full_stage_machine_then_resume_is_a_no_op(self):
        b = make_batch(self.root)
        scr = Screen({})
        launches = []
        res = self.run_prep(b, scr, launches)
        self.assertEqual(res["prepared"], [TASK], res)
        self.assertEqual(res["failed"], [])
        # 2 fake runs (one per bundle: parity proof + the request to freeze), then 2 hinted + 1 control
        self.assertEqual(len(launches), 5, [x["job_name"] for x in launches])
        fakes = [x for x in launches if "fake-" in x["job_name"]]
        self.assertEqual(sorted(x["job_name"].rsplit("-", 1)[-1] for x in fakes), ["control", "hinted"])
        self.assertTrue(all(x["fake"] for x in fakes))
        real = [x for x in launches if "fake-" not in x["job_name"]]
        kinds = sorted(x["job_name"].rsplit("-", 1)[-1] for x in real)
        self.assertEqual(kinds, ["control", "hinteda", "hintedb"])
        hinted = [x for x in real if x["job_name"].endswith(("hinteda", "hintedb"))]
        self.assertTrue(all(x["freeze_first_request"] for x in hinted))
        self.assertTrue(all(x["midrun_snapshot_every"] == 1200 for x in hinted))
        control = [x for x in real if x["job_name"].endswith("control")][0]
        self.assertEqual(control["midrun_snapshot_every"], 0)
        self.assertEqual(Path(control["bundle"]).name, f"{TASK}-s100-control")
        snap = json.loads((self.root / "state" / "snapshot-sum-task-s100.json").read_text())
        self.assertEqual(snap["snapshot_id"], "snap-1")
        st = json.loads((self.root / "batch" / "r1" / "state.json").read_text())[TASK]
        self.assertEqual(st["stage"], "launched")
        self.assertEqual(len(st["runs"]), 3)
        # resume: everything cached, nothing rebuilt or relaunched
        scr2 = Screen({})
        res2 = self.run_prep(b, scr2, [])
        self.assertEqual(res2["prepared"], [TASK])
        self.assertEqual(scr2.calls, [])


    def test_not_resumable_unit_is_never_prepared_or_launched(self):
        b = make_batch(self.root)
        b.state.update(TASK, stage="not_resumable", not_resumable_reason="reviewed: replay diverges")
        scr, launches = Screen({}), []
        res = self.run_prep(b, scr, launches)
        self.assertEqual(res["prepared"], [])
        self.assertEqual(res["not_resumable"], [TASK])
        self.assertEqual(launches, [])
        self.assertEqual(scr.calls, [])
        self.assertEqual(b.state.unit(TASK)["stage"], "not_resumable")

    def test_reviewed_exceptions_sidecar_is_merged_into_every_bundle(self):
        side = self.root / "batch" / "r1" / "units" / TASK / "replay_exceptions.json"
        side.parent.mkdir(parents=True, exist_ok=True)
        review = {"call_00_a": {"kind": "reviewed_masked_output", "mask_version": 1, "order": "ordered",
                                "command_sha256": "1" * 64, "changed_numbers_reused_later": False, "note": "timings"}}
        side.write_text(json.dumps(review))
        b = make_batch(self.root)
        self.run_prep(b, Screen({}), [])
        for kind in ("hint", "control"):
            meta = json.loads((b.unit_dir(TASK) / "bundles" / f"{TASK}-s100-{kind}" / "bundle.json").read_text())
            self.assertEqual(meta["replay_reviewed_exceptions"], review)
        # a changed review set invalidates the fake parity proof: the cached fake was made against another identity
        review["call_00_b"] = dict(review["call_00_a"], command_sha256="2" * 64)
        side.write_text(json.dumps(review))
        launches = []
        self.run_prep(b, Screen({}), launches)
        fakes = [x["job_name"] for x in launches if "fake-" in x["job_name"]]
        self.assertTrue(fakes, launches)
        self.assertTrue(all("-v2" in name for name in fakes), fakes)
        for kind in ("hint", "control"):
            meta = json.loads((b.unit_dir(TASK) / "bundles" / f"{TASK}-s100-{kind}" / "bundle.json").read_text())
            self.assertEqual(meta["replay_reviewed_exceptions"], review)

    def test_blocked_cut_moves_inside_the_fork_window(self):
        b = make_batch(self.root)
        scr = Screen({100: "block", 98: "block", 102: "block", 96: "ok"})
        res = self.run_prep(b, scr, [])
        self.assertEqual(res["prepared"], [TASK])
        st = b.state.unit(TASK)
        self.assertEqual(st["cut_line"], 96)
        self.assertTrue(st["cut_moved"])
        self.assertEqual(st["screen_verdict"], "ok")
        screened = [c for call in scr.calls for c in call]
        self.assertIn(96, screened)
        self.assertTrue(all(c < 140 for c in screened))  # never at or past the fork
        # nearest first, earlier before later
        self.assertEqual(screened[:3], [100, 96, 104])

    def test_review_only_cut_is_taken_with_its_verdict(self):
        b = make_batch(self.root)
        res = self.run_prep(b, Screen({c: "review" for c in range(76, 125)}), [])
        self.assertEqual(res["prepared"], [TASK])
        self.assertEqual(b.state.unit(TASK)["screen_verdict"], "review")

    def test_everything_blocking_fails_the_unit_only(self):
        b = make_batch(self.root)
        res = self.run_prep(b, Screen({c: "block" for c in range(70, 145)}), [])
        self.assertEqual(res["prepared"], [])
        self.assertEqual(res["failed"], [TASK])
        st = b.state.unit(TASK)
        self.assertEqual(st["stage"], "failed")
        self.assertIn("blocks", st["error"])

    def test_changed_hint_rebuilds_the_bundles(self):
        b = make_batch(self.root)
        launches = []
        self.run_prep(b, Screen({}), launches)
        first = json.loads((self.root / "batch" / "r1" / "units" / TASK / "bundles.json").read_text())
        (self.root / "hint.txt").write_text(HINT + " Also: report the median.")
        b2 = make_batch(self.root)
        launches2 = []
        self.run_prep(b2, Screen({}), launches2)
        second = json.loads((self.root / "batch" / "r1" / "units" / TASK / "bundles.json").read_text())
        self.assertNotEqual(first["inputs"], second["inputs"])
        self.assertTrue(second["rebuilt_for_input_change"])

    def test_duplicate_task_rows_are_refused(self):
        (self.root / "manifest.csv").unlink(missing_ok=True)
        manifest(self.root, [{"task": TASK, "group": "g", "source_trial": f"{TASK}__abc", "cut_line": 100,
                              "after_compaction": "True", "n_forks": "1", "class": "single_fork", "hint_path": "",
                              "leak_ok": "True"},
                             {"task": TASK, "group": "g", "source_trial": f"{TASK}__abc", "cut_line": 96,
                              "after_compaction": "True", "n_forks": "1", "class": "single_fork", "hint_path": "",
                              "leak_ok": "True"}])
        with self.assertRaises(SystemExit):
            make_batch(self.root)

    def test_missing_parity_reference_fails_before_launch(self):
        b = make_batch(self.root)

        def no_parity(**kw):
            return fake_build(**{**kw, "cut_line": kw["cut_line"]}) if False else (
                Path(kw["out"]) / kw["run_id"], {"cut_line": kw["cut_line"], "orig_request_at_k": None,
                                                 "orig_request_locate_error": "no model-proxy log"})

        with mock.patch("agent_snapshot_resume.bundle.resolve_agent_timeout", lambda *a, **k: 600.0), \
                mock.patch("agent_snapshot_resume.screen.screen_trial", Screen({})), \
                mock.patch("agent_snapshot_resume.bundle.build", no_parity):
            out = b.prep({"task": TASK, "cut_line": 100, "hint_path": str(self.root / "hint.txt"), "group": "g"})
        self.assertIn("error", out)
        self.assertIn("no original request", out["error"])
        self.assertEqual(b.state.unit(TASK)["stage"], "failed")


class Status(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.b = make_batch(self.root)

    def trial(self, job, *, result=None, requests=None, verifier=None):
        d = self.root / "jobs" / job / f"{TASK}__xyz"
        (d / "agent" / "resume-proxy" / "run-1").mkdir(parents=True)
        if result is not None:
            (d / "result.json").write_text(json.dumps(result))
        if requests:
            (d / "agent" / "resume-proxy" / "run-1" / "requests.jsonl").write_text(
                "\n".join(json.dumps(r) for r in requests) + "\n")
        if verifier:
            (d / "verifier").mkdir()
            (d / "verifier" / "test-stdout.txt").write_text(verifier)
        return d

    def test_done_run_reports_reward_cost_hints_and_failures(self):
        job = self.b.job({"task": TASK, "cut_line": 100}, "hinteda")
        self.trial(job, result={"verifier_result": {"rewards": {"reward": 1.0}},
                                "agent_result": {"cost_usd": 1.23}, "exception_info": None},
                   requests=[{"n": 0, "hint_chain": [{"blocks": 1, "occurrences": 1}], "compact_summary": False},
                             {"n": 1, "hint_chain": [{"blocks": 1, "occurrences": 1}], "compact_summary": True}],
                   verifier="PASSED test_a\nFAILED test_b\n")
        self.b.state.update(TASK, stage="launched", cut_line=100,
                            runs={job: {"run": "hinteda", "kind": "hinted", "job": job,
                                        "trial": f"{TASK}__xyz", "launched": "now"}})
        row = self.b.status()[0]
        self.assertEqual(row["state"], "done")
        self.assertEqual(row["reward"], 1.0)
        self.assertEqual(row["cost_usd"], 1.23)
        self.assertEqual(row["hint_counts"], [1])
        self.assertEqual(row["n_compactions"], 1)
        self.assertEqual(row["failed_tests"], ["FAILED test_b"])

    def test_running_and_missing_results(self):
        job = self.b.job({"task": TASK, "cut_line": 100}, "control")
        self.b.state.update(TASK, stage="launched", cut_line=100,
                            runs={job: {"run": "control", "kind": "control", "job": job, "launched": "now"}})
        self.assertEqual(self.b.status()[0]["state"], "running")
        self.assertEqual(self.b.status()[0]["class"], None)

    def test_infra_vs_model_failure_class(self):
        for job, exc, want in ((self.b.job({"task": TASK, "cut_line": 100}, "hinteda"), "SandboxConnectError: 502", "infra"),
                               (self.b.job({"task": TASK, "cut_line": 100}, "hintedb"), "agent turn limit", "model")):
            self.trial(job, result={"verifier_result": {}, "agent_result": {}, "exception_info": exc})
        runs = {self.b.job({"task": TASK, "cut_line": 100}, r): {"run": r, "kind": "hinted", "job": "",
                                                                 "trial": f"{TASK}__xyz", "launched": "now"}
                for r in ("hinteda", "hintedb")}
        self.b.state.update(TASK, stage="launched", cut_line=100, runs=runs)
        got = {r["run"]: r["class"] for r in self.b.status()}
        self.assertEqual(got, {"hinteda": "infra", "hintedb": "model"})


class StateFile(unittest.TestCase):
    def test_atomic_and_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = batch.State(Path(tmp))
            s.update("t", stage="screen", cut_line=5)
            s.update("t", stage="bundles")
            on_disk = json.loads((Path(tmp) / "state.json").read_text())
            self.assertEqual(on_disk["t"]["stage"], "bundles")
            self.assertEqual(on_disk["t"]["cut_line"], 5)
            events = (Path(tmp) / "events.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(e)["stage"] for e in events], ["screen", "bundles"])
            self.assertFalse(list(Path(tmp).glob("*.tmp")))
