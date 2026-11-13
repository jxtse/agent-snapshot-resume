"""Manifest-driven batch driver: screen -> bundles -> template -> snapshot -> fake parity -> real launches.

One *unit* is one task's cut (a manifest row). Its stages, all idempotent and resumable:

    screen    offline replay-risk report at the cut (and, if it blocks, at nearby cuts inside the fork window)
    bundles   --mode hint bundle (all hinted replicates share it) + --mode same control bundle
    template  snapshot-capable template of the task image (one per task; reused everywhere)
    snapshot  replay steps < cut once and snapshot it (shared by every bundle of that cut identity)
    fake      --fake launch of each bundle: first-request parity gate + the request to freeze
    launch    the real runs: --hinted-reps hinted, --control-reps control, concurrency capped

State lives in ``<batch-dir>/state.json`` (atomic replace) plus ``events.jsonl`` (append-only), so an interrupted
batch resumes where it stopped instead of rebuilding or relaunching. A launched run is identified by its job name and
is *never* relaunched once ``runs/<job>/job.json`` exists.

Usage:
    asr batch run    --name o70-r1 --manifest round1/manifest.csv --work-dir .
    asr batch prep   --name o70-r1 --manifest ... (stops before the real launches)
    asr batch launch --name o70-r1 (launches whatever is already prepared)
    asr batch status --name o70-r1 [--json]
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

STAGES = ("screen", "bundles", "template", "snapshot", "fake", "launch")
TERMINAL_FAIL = ("screen_block", "bundle_error", "template_error", "snapshot_error", "fake_parity_failed",
                 "launch_error")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row = {k: (v or "").strip() for k, v in row.items()}
            row["cut_line"] = int(row["cut_line"])
            row["row"] = i
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    dup = sorted({r["task"] for r in rows if [x["task"] for x in rows].count(r["task"]) > 1})
    if dup:
        # one unit per task: two rows for the same task would share one unit key and launch the wrong bundle
        raise SystemExit(f"{path}: duplicate task rows {dup}; split them into separate batches")
    return rows


class State:
    """Unit state in one JSON file, written atomically; every mutation is also appended to events.jsonl."""

    def __init__(self, batch_dir: Path):
        self.dir = Path(batch_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"
        self.events = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self.units: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            self.units = json.loads(self.path.read_text())

    def unit(self, task: str) -> dict[str, Any]:
        with self._lock:
            return self.units.setdefault(task, {"task": task, "stage": "new", "runs": {}, "errors": []})

    def update(self, task: str, **kw) -> dict[str, Any]:
        with self._lock:
            u = self.units.setdefault(task, {"task": task, "stage": "new", "runs": {}, "errors": []})
            u.update(kw)
            u["updated"] = _now()
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.units, indent=1, sort_keys=True))
            os.replace(tmp, self.path)
            with self.events.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": _now(), "task": task, "stage": u.get("stage"),
                                    **{k: v for k, v in kw.items() if k not in ("runs",)}}, ensure_ascii=False) + "\n")
            return u


class Batch:
    def __init__(self, *, name: str, work_dir: Path, manifest: Path, tasks_dir: Path, jobs_dir: Path,
                 runs_dir: Path, state_dir: Path, agent_env: dict[str, str] | None = None,
                 passthrough_args: list[str] | None = None, model: str | None = None,
                 upstream_url: str | None = None, hinted_reps: int = 2, control_reps: int = 1,
                 midrun_snapshot_every: float = 1200.0, cut_window: int = 200, max_cut_candidates: int = 16,
                 prep_concurrency: int = 6,
                 launch_concurrency: int = 20, only: list[str] | None = None, agent: str = "claude-code",
                 verifier: bool = True, log=print):
        self.name, self.work = name, Path(work_dir)
        self.batch_dir = self.work / "batch" / name
        self.manifest_path = Path(manifest)
        self.rows = load_manifest(self.manifest_path)
        if only:
            self.rows = [r for r in self.rows if r["task"] in only]
        self.tasks_dir = Path(tasks_dir)
        self.jobs_dir, self.runs_dir, self.state_dir = Path(jobs_dir), Path(runs_dir), Path(state_dir)
        self.agent_env, self.passthrough_args, self.model = agent_env or {}, passthrough_args, model
        self.upstream_url = upstream_url
        self.hinted_reps, self.control_reps = hinted_reps, control_reps
        self.midrun_every = midrun_snapshot_every
        self.cut_window = cut_window
        self.max_cut_candidates = max_cut_candidates
        self.prep_concurrency, self.launch_concurrency = prep_concurrency, launch_concurrency
        self.agent_name, self.verifier = agent, verifier
        self.state = State(self.batch_dir)
        self.log = log
        self._adapter = None

    # ------------------------------------------------------------------ helpers
    @property
    def adapter(self):
        from .agents import get_adapter
        if self._adapter is None:
            self._adapter = get_adapter(self.agent_name)
        return self._adapter

    def unit_dir(self, task: str) -> Path:
        d = self.batch_dir / "units" / task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def source(self, row: dict[str, Any]) -> dict[str, Any]:
        """The L0 trial this cut comes from (round1/<task>/source.json next to the manifest wins)."""
        cand = self.manifest_path.parent / row["task"] / "source.json"
        if cand.is_file():
            src = json.loads(cand.read_text())
            if src.get("trial_dir"):
                return src
        return {"trial_dir": row.get("source_dir") or "", "trial": row.get("source_trial")}

    def source_session_sha(self, row: dict[str, Any]) -> str | None:
        """Session hash of the source trial (part of the bundle input identity)."""
        src = self.source(row)
        p = Path(src["trial_dir"])
        js = p / "agent" / "sessions" / "projects"
        cand = sorted(js.glob("*/*.jsonl"))
        if not cand:
            return None
        return __import__("hashlib").sha256(cand[-1].read_bytes()).hexdigest()

    def task_dir(self, row: dict[str, Any]) -> Path:
        return self.tasks_dir / row["task"]

    def job(self, row: dict[str, Any], run: str) -> str:
        return f"{self.name}-{row['task']}-s{row['cut_line']}-{run}"

    # ------------------------------------------------------------------ stage: screen
    def stage_screen(self, row: dict[str, Any]) -> dict[str, Any]:
        from . import screen
        from .bundle import resolve_agent_timeout

        task = row["task"]
        ud = self.unit_dir(task)
        out = ud / "screen.json"
        if out.is_file():
            rep = json.loads(out.read_text())
            if rep.get("manifest_cut") == row["cut_line"]:
                if rep.get("cut_line") is None:
                    raise RuntimeError(f"every cut in the fork window blocks ({len(rep.get('candidates') or [])} "
                                       "candidates): see screen.json")
                return rep
        src = self.source(row)
        trial_dir = Path(src["trial_dir"])
        if not trial_dir.is_dir():
            raise RuntimeError(f"source trial dir missing: {trial_dir}")
        if not (self.task_dir(row) / "task.toml").is_file():
            # no template, no healthcheck, no agent timeout: this task cannot be resumed here at all
            raise RuntimeError(f"no local task dir ({self.task_dir(row)}): not resumable")
        fork_line = None
        fork = self.manifest_path.parent / task / "fork.json"
        if fork.is_file():
            fj = json.loads(fork.read_text())
            lines = [f["session_line"] for f in fj.get("forks") or [] if f.get("session_line")]
            fork_line = max(lines) if lines else None
        # a cut must start an agent step: snap the manifest cut to the nearest one, then take the neighbourhood
        ir = self.adapter.load_trace(trial_dir, agent_timeout_s=resolve_agent_timeout(trial_dir, None,
                                                                                     self.task_dir(row)),
                                    task=task)
        if ir.get("error"):
            raise RuntimeError(f"{trial_dir.name}: {ir['error']}")
        starts = sorted({st["line_range"][0] for st in ir["steps"] if st["source"] == "agent"})
        if not starts:
            raise RuntimeError("no agent steps in the session")
        near = min(starts, key=lambda s: (abs(s - row["cut_line"]), -s))
        pool = [s for s in starts if abs(s - near) <= self.cut_window and (fork_line is None or s < fork_line)]
        # nearest first, then earlier before later (a cut moved forward risks crossing the wrong decision)
        cands = [near] + sorted((s for s in pool if s != near), key=lambda s: (abs(s - near), s))
        by_cut: dict[int, dict[str, Any]] = {}
        skipped: list[dict[str, Any]] = []
        for c in cands[:self.max_cut_candidates]:
            try:
                rep = screen.screen_trial(self.adapter, trial_dir, cut_lines=[c], task_dir=self.task_dir(row), ir=ir)
            except (SystemExit, ValueError) as exc:  # e.g. a step at index 0 has no prefix to replay
                skipped.append({"cut_line": c, "error": str(exc)[:200]})
                continue
            if rep:
                by_cut[c] = rep[0]
        chosen = None
        for c in cands:
            r = by_cut.get(c)
            if r and r.get("verdict") == "ok":
                chosen = c
                break
        if chosen is None:
            for c in cands:
                r = by_cut.get(c)
                if r and r.get("verdict") == "review":
                    chosen = c
                    break
        rep = {"task": task, "manifest_cut": row["cut_line"], "snapped_cut": near, "fork_session_line": fork_line,
               "candidates": [screen.summarize(by_cut[c]) for c in cands if c in by_cut], "skipped": skipped,
               "cut_line": chosen, "moved": chosen != row["cut_line"],
               "verdict": (by_cut.get(chosen) or {}).get("verdict") if chosen else "block",
               "reports": {str(c): by_cut[c] for c in cands if c in by_cut}, "ts": _now()}
        out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        if chosen is None:
            raise RuntimeError(f"every cut in the fork window blocks ({len(by_cut)} screened, {len(skipped)} skipped)")
        return rep

    # ------------------------------------------------------------------ stage: bundles
    def stage_bundles(self, row: dict[str, Any], cut: int) -> dict[str, Any]:
        from . import bundle as bundle_mod

        task = row["task"]
        ud = self.unit_dir(task)
        hint = Path(row["hint_path"]).read_text().strip()
        src = self.source(row)
        specs = {"hinted": {"mode": "hint", "hint": hint, "run_id": f"{task}-s{cut}-hint"},
                 "control": {"mode": "same", "hint": None, "run_id": f"{task}-s{cut}-control"}}
        # A cached bundle is only reusable if it was built from the same inputs; a changed hint file, cut or source
        # session must rebuild it, and a downstream fake/parity proof must never legitimise the stale one.
        inputs = {"cut_line": cut, "source_trial_dir": str(Path(src["trial_dir"]).resolve()),
                  "hint_sha256": hashlib.sha256(hint.encode()).hexdigest() if hint else None,
                  "session_sha256": self.source_session_sha(row)}
        cached = {}
        if (ud / "bundles.json").is_file():
            try:
                cached = json.loads((ud / "bundles.json").read_text())
            except ValueError:
                cached = {}
        rebuild = bool(cached) and cached.get("inputs") != inputs
        out: dict[str, Any] = {"cut_line": cut, "inputs": inputs, "bundles": {},
                               "rebuilt_for_input_change": rebuild}
        for kind, spec in specs.items():
            bdir = ud / "bundles" / spec["run_id"]
            if bdir.is_dir() and (bdir / "bundle.json").is_file() and not rebuild:
                meta = json.loads((bdir / "bundle.json").read_text())
            else:
                bdir, meta = bundle_mod.build(adapter=self.adapter, trial_dir=Path(src["trial_dir"]),
                                             atif_step=None, cut_line=cut, out=ud / "bundles",
                                             task_dir=self.task_dir(row), mode=spec["mode"], hint=spec["hint"],
                                             run_id=spec["run_id"], persist_hint=(spec["mode"] == "hint"),
                                             force=True)
            out["bundles"][kind] = {"dir": str(bdir), "run_id": spec["run_id"], "cut_line": meta.get("cut_line"),
                                    "agent_budget_s": meta.get("agent_budget_s"),
                                    "n_replay_actions": meta.get("n_replay_actions"),
                                    "parity_located": bool(meta.get("orig_request_at_k")),
                                    "parity_error": meta.get("orig_request_locate_error")}
            if not meta.get("orig_request_at_k"):
                raise RuntimeError(f"{kind} bundle has no original request at the cut: "
                                   f"{meta.get('orig_request_locate_error')}")
        (ud / "bundles.json").write_text(json.dumps(out, indent=1))
        return out

    # ------------------------------------------------------------------ stage: template
    def stage_template(self, row: dict[str, Any]) -> dict[str, Any]:
        from .sandbox import templates

        task = row["task"]
        out = self.state_dir / f"template-{task}.json"
        rec = json.loads(out.read_text()) if out.is_file() else {}
        if "ready" not in str(rec.get("status")):
            rec = templates.build_template(task_dir=self.task_dir(row), state_dir=self.state_dir)
        if "ready" not in str(rec.get("status")):
            raise RuntimeError(f"template build failed: {rec.get('status')} {rec.get('reason')}")
        return rec

    # ------------------------------------------------------------------ stage: snapshot
    def stage_snapshot(self, row: dict[str, Any], cut: int, bundled: dict[str, Any],
                       tmpl_id: str | None = None) -> dict[str, Any]:
        from .sandbox import source

        task = row["task"]
        tag = f"{task}-s{cut}"
        if not tmpl_id:
            raise RuntimeError("no ready template id for this task")
        out = self.state_dir / f"snapshot-{tag}.json"
        if out.is_file():
            rec = json.loads(out.read_text())
            if rec.get("replay_gate_passed") and rec.get("snapshot_id"):
                return rec
            raise RuntimeError(f"snapshot state {out} exists without a passed replay gate; inspect and use a new tag")
        rec = source.prepare_source(adapter=self.adapter, task_dir=self.task_dir(row),
                                    bundle=Path(bundled["bundles"]["hinted"]["dir"]),
                                    template=str(tmpl_id), tag=tag, state_dir=self.state_dir)
        if not rec.get("snapshot_id"):
            raise RuntimeError("prepare-snapshot produced no snapshot id")
        return rec

    # ------------------------------------------------------------------ stage: fake parity
    def stage_fake(self, row: dict[str, Any], cut: int, bundled: dict[str, Any], snap_path: Path) -> dict[str, Any]:
        from .launch import launch_one

        task = row["task"]
        ud = self.unit_dir(task)
        out: dict[str, Any] = {"runs": {}}
        cache_path = ud / "fake.json"
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        for kind, b in bundled["bundles"].items():
            job = f"{self.job(row, 'fake-' + kind)}"
            cached = (cache.get("runs") or {}).get(kind) or {}
            if cached.get("parity_ok") and Path(cached.get("first_request", "")).is_file():
                out["runs"][kind] = {**cached, "cached": True}
                continue
            rc = launch_one(adapter=self.adapter, bundle=Path(b["dir"]), task_dir=self.task_dir(row), job_name=job,
                            jobs_dir=self.jobs_dir, runs_dir=self.runs_dir, snapshot_state=snap_path, fake=True,
                            verifier=False, model=self.model, extra_env=self.agent_env,
                            original_trial_dir=self.source(row).get("trial_dir"),
                            passthrough_args=self.passthrough_args, parity_gate="require",
                            midrun_snapshot_every=0)
            trial = self.find_trial(job)
            if trial is None:
                raise RuntimeError(f"fake run {job} produced no trial dir (rc={rc})")
            proxies = sorted((trial / "agent" / "resume-proxy").glob("*/"))
            if not proxies:
                raise RuntimeError(f"fake run {job} wrote no proxy log")
            first = proxies[-1] / "first_request_rewritten.json"
            parity = proxies[-1] / "first_request_parity.json"
            if not first.is_file() or not parity.is_file():
                raise RuntimeError(f"fake run {job} missed {first.name}/{parity.name}")
            ok = bool(json.loads(parity.read_text()).get("ok"))
            (ud / f"fake-parity-{kind}.json").write_text(parity.read_text())  # record for the batch, not for freezing
            if not ok:
                raise RuntimeError(f"fake parity failed for {kind}: {parity}")
            # --freeze-first-request resolves the parity report and the bundle identity *next to* this file, so pass
            # the request where the fake run wrote it (inside the trial's resume-proxy log), never a copy.
            out["runs"][kind] = {"job": job, "first_request": str(first), "parity_ok": True, "trial": str(trial),
                                 "parity_report": str(parity)}
        (ud / "fake.json").write_text(json.dumps(out, indent=1))
        return out

    def find_trial(self, job: str) -> Path | None:
        d = self.jobs_dir / job
        if not d.is_dir():
            return None
        cands = [p for p in d.iterdir() if p.is_dir()]
        return cands[0] if cands else None

    # ------------------------------------------------------------------ stage: launch
    def launch_runs(self, row: dict[str, Any], cut: int, bundled: dict[str, Any], snap_path: Path,
                    fake: dict[str, Any]) -> list[dict[str, Any]]:
        from .launch import launch_one

        task = row["task"]
        plan = []
        for kind, reps in (("hinted", self.hinted_reps), ("control", self.control_reps)):
            for i in range(reps):
                run = kind if reps == 1 else f"{kind}{'abc'[i] if i < 3 else i + 1}"
                plan.append({"task": task, "run": run, "kind": kind, "cut_line": cut,
                             "job": self.job(row, run), "bundle": bundled["bundles"][kind]["dir"],
                             "freeze": fake["runs"][kind]["first_request"], "snapshot_state": str(snap_path)})
        launched = []
        for item in plan:
            u = self.state.unit(task)
            already = (self.runs_dir / item["job"] / "job.json").is_file() or self.find_trial(item["job"])
            if already:
                # the run dir (written before Harbor starts) and the trial dir are the markers: a lost state entry
                # must never cause a second launch of the same job
                runs = u.get("runs") or {}
                if item["job"] not in runs:
                    trials = [p.name for p in (self.jobs_dir / item["job"]).glob("*") if p.is_dir()] \
                        if (self.jobs_dir / item["job"]).is_dir() else []
                    runs[item["job"]] = {"run": item["run"], "kind": item["kind"], "job": item["job"],
                                         "trial": (trials[-1] if trials else None), "rc": None,
                                         "launched": "(recovered from the run dir)"}
                    self.state.update(task, runs=runs)
                launched.append({**item, "cached": True})
                continue
            rc = launch_one(adapter=self.adapter, bundle=Path(item["bundle"]), task_dir=self.task_dir(row),
                            job_name=item["job"], jobs_dir=self.jobs_dir, runs_dir=self.runs_dir,
                            snapshot_state=snap_path, fake=False, verifier=self.verifier, model=self.model,
                            upstream_url=self.upstream_url, extra_env=self.agent_env,
                            original_trial_dir=self.source(row).get("trial_dir"),
                            passthrough_args=self.passthrough_args, parity_gate="require",
                            freeze_first_request=item["freeze"],
                            midrun_snapshot_every=self.midrun_every if item["kind"] == "hinted" else 0)
            trials = [p.name for p in (self.jobs_dir / item["job"]).glob("*") if p.is_dir()] \
                if (self.jobs_dir / item["job"]).is_dir() else []
            runs = self.state.unit(task).get("runs") or {}
            runs[item["job"]] = {"run": item["run"], "kind": item["kind"], "job": item["job"],
                                 "trial": (trials[0] if trials else None), "rc": rc, "launched": _now()}
            self.state.update(task, runs=runs)
            launched.append({**item, "rc": rc, "trial": trials[0] if trials else None})
        return launched

    # ------------------------------------------------------------------ driver
    def prep(self, row: dict[str, Any]) -> dict[str, Any]:
        task = row["task"]
        try:
            self.state.update(task, stage="screen", group=row.get("group"))
            scr = self.stage_screen(row)
            cut = int(scr["cut_line"])
            self.state.update(task, stage="bundles", cut_line=cut, screen_verdict=scr["verdict"],
                              cut_moved=scr["moved"], fork_session_line=scr["fork_session_line"])
            bundled = self.stage_bundles(row, cut)
            self.state.update(task, stage="template")
            tmpl = self.stage_template(row)
            self.state.update(task, stage="snapshot", template_id=tmpl.get("template_id"))
            snap = self.stage_snapshot(row, cut, bundled, tmpl.get("template_id"))
            snap_path = self.state_dir / f"snapshot-{task}-s{cut}.json"
            self.state.update(task, stage="fake", snapshot_id=snap["snapshot_id"])
            fake = self.stage_fake(row, cut, bundled, snap_path)
            self.state.update(task, stage="ready")
            return {"task": task, "cut_line": cut, "bundled": bundled, "snap": snap_path, "fake": fake}
        except BaseException as exc:  # noqa: BLE001 - one unit must never take the batch down
            stage = self.state.unit(task).get("stage", "?")
            err = f"{stage}: {exc!r}"[:900]
            self.state.update(task, stage="failed", failed_stage=stage, error=err)
            self.log(f"[{self.name}] {task} FAILED at {stage}: {exc!r}"[:600])
            return {"task": task, "error": err}

    def run(self, *, do_launch: bool = True) -> dict[str, Any]:
        rows = {r["task"]: r for r in self.rows}
        prep_q: queue.Queue = queue.Queue()
        launch_q: queue.Queue = queue.Queue()
        results: dict[str, Any] = {"prepared": [], "failed": [], "launched": [], "launch_failed": []}
        lock = threading.Lock()
        for r in self.rows:
            st = self.state.unit(r["task"])
            if st.get("stage") == "failed":
                results["failed"].append(r["task"])
            else:
                prep_q.put(r)

        def prep_worker():
            while True:
                row = prep_q.get()
                if row is None:
                    prep_q.task_done()
                    return
                out = self.prep(row)
                with lock:
                    if "error" in out:
                        results["failed"].append(out["task"])
                    else:
                        results["prepared"].append(out["task"])
                        if do_launch:
                            launch_q.put(out)
                prep_q.task_done()

        def launch_worker():
            while True:
                item = launch_q.get()
                if item is None:
                    return
                row = rows[item["task"]]
                try:
                    got = self.launch_runs(row, item["cut_line"], item["bundled"], item["snap"], item["fake"])
                    with lock:
                        results["launched"].extend(got)
                    self.state.update(item["task"], stage="launched", launched=[g["job"] for g in got])
                except BaseException as exc:  # noqa: BLE001
                    with lock:
                        results["launch_failed"].append({"task": item["task"], "error": repr(exc)[:500]})
                    self.state.update(item["task"], stage="failed", failed_stage="launch", error=repr(exc)[:900])
                    self.log(f"[{self.name}] {item['task']} LAUNCH FAILED: {exc!r}"[:600])

        threads = [threading.Thread(target=prep_worker, daemon=True) for _ in range(max(1, self.prep_concurrency))]
        for t in threads:
            t.start()
        for _ in threads:
            prep_q.put(None)  # one sentinel per worker: it exits after the queue is drained
        lthreads = []
        if do_launch:
            lthreads = [threading.Thread(target=launch_worker, daemon=True)
                        for _ in range(max(1, self.launch_concurrency))]
            for t in lthreads:
                t.start()
        for t in threads:
            t.join()
        if lthreads:
            for _ in lthreads:
                launch_q.put(None)  # prep is done: no more work can arrive
            for t in lthreads:
                t.join()
        return results

    # ------------------------------------------------------------------ status
    def status(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.rows:
            task = row["task"]
            st = self.state.unit(task)
            runs = dict(st.get("runs") or {})
            # the run dirs are the source of truth for "this was launched": a lost state entry must not hide a run
            for job_dir in sorted(self.runs_dir.glob(f"{self.name}-{task}-s*-*")):
                job = job_dir.name
                if job in runs or "fake" in job:
                    continue
                trials = [p.name for p in (self.jobs_dir / job).glob("*") if p.is_dir()] \
                    if (self.jobs_dir / job).is_dir() else []
                runs[job] = {"run": job.rsplit("-", 1)[-1], "kind": "hinted" if "control" not in job else "control",
                             "job": job, "trial": (trials[-1] if trials else None), "launched": "(from the run dir)"}
            for job, info in sorted(runs.items()):
                rows.append(self.run_status(row, st, job, info))
            if not runs:
                rows.append({"task": task, "run": None, "job": None, "stage": st.get("stage"),
                             "state": st.get("stage"), "cut_line": st.get("cut_line"), "error": st.get("error")})
        return rows

    def run_status(self, row: dict[str, Any], st: dict[str, Any], job: str, info: dict[str, Any]) -> dict[str, Any]:
        trial = self.jobs_dir / job / (info.get("trial") or "")
        out: dict[str, Any] = {"task": row["task"], "run": info.get("run"), "kind": info.get("kind"), "job": job,
                               "cut_line": st.get("cut_line"), "trial_dir": str(trial),
                               "state": "launched", "reward": None, "failed_tests": [], "cost_usd": None,
                               "hint_counts": [], "n_compactions": 0, "class": None, "error": None}
        res_path = trial / "result.json"
        if not res_path.is_file():
            started = info.get("launched")
            out["state"] = "running" if started else "pending"
            return out
        try:
            res = json.loads(res_path.read_text())
        except ValueError:
            out["state"] = "running"  # still being written
            return out
        out["state"] = "done"
        out["reward"] = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        out["cost_usd"] = (res.get("agent_result") or {}).get("cost_usd")
        exc = res.get("exception_info")
        if exc:
            out["error"] = str(exc)[:400]
            out["class"] = "infra" if any(k in str(exc).lower() for k in
                                          ("sandbox", "snapshot", "connect", "timeout connecting", "http")) else "model"
        out["failed_tests"] = failed_tests(trial)
        reqs = sorted((trial / "agent" / "resume-proxy").glob("*/requests.jsonl"))
        if reqs:
            recs = [json.loads(l) for l in reqs[-1].read_text().splitlines() if l.strip()]
            main = [r for r in recs if r.get("n") is not None]
            out["n_requests"] = len(main)
            out["n_compactions"] = sum(1 for r in main if r.get("compact_summary"))
            if main:
                out["hint_counts"] = [c.get("blocks") for c in (main[0].get("hint_chain") or [])] or \
                                     [main[0].get("hint_blocks")]
                out["hint_counts_last"] = [c.get("blocks") for c in (main[-1].get("hint_chain") or [])] or \
                                          [main[-1].get("hint_blocks")]
        gate = trial / "agent" / "resume-proxy"
        if gate.is_dir():
            for g in gate.glob("*/first_request_parity.json"):
                try:
                    out["first_request_parity"] = bool(json.loads(g.read_text()).get("ok"))
                except ValueError:
                    pass
        if out["reward"] is None and not out["error"]:
            out["class"] = out["class"] or "no_reward"
        return out


def failed_tests(trial: Path, limit: int = 12) -> list[str]:
    """Test ids the verifier reported as failing (from its stdout), without dumping the log."""
    out: list[str] = []
    for p in sorted(trial.glob("verifier/*stdout*")):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("FAILED ") or s.startswith("FAIL: ") or " FAILED" in s[:40]:
                out.append(s[:160])
                if len(out) >= limit:
                    return out
    return out


def print_status(rows: list[dict[str, Any]]) -> None:
    cols = ("task", "run", "state", "reward", "cost_usd", "hint_counts", "n_compactions", "class", "error")
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r.get(c) is None else str(r.get(c))[:70] for c in cols))
