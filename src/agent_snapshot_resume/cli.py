"""`asr` command line. Run `asr <command> --help` for details.

  build-bundle      Harbor trial dir + step k -> resume bundle (offline)
  build-template    register a snapshot-capable sandbox template from a task image
  prepare-snapshot  source sandbox: install agent, replay steps < k, gate, fingerprint, snapshot
  launch            write the Harbor job config (and run Harbor unless --dry-run); snapshots the live clone
  snapshot-midrun   snapshot a running job's live clones (periodic + after every compaction)
  equiv             check a rewritten first request against the original session prefix (offline)
  parity            check a rewritten first request against the original run's request at the cut (offline)
  screen            replayability risks of candidate cuts (offline, no sandbox)
  selfcheck         simulate the resumed first request from a bundle, rewrite it, run equiv (offline)
  sandboxes         list sandboxes tagged with ASR_OWNER_TAG; --kill-id to kill one
  delete-snapshot   delete a snapshot
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from . import config


def _adapter(name):
    from .agents import get_adapter
    return get_adapter(name)


def _load_snapshot_record(path):
    """A mid-run snapshot record: one JSON object, or a .jsonl file (its last line)."""
    text = Path(path).read_text().strip()
    if text.startswith("{"):
        return json.loads(text)
    lines = [l for l in text.splitlines() if l.strip()]
    return json.loads(lines[-1])


def cmd_build_bundle(a):
    from .bundle import build

    hint = Path(a.hint_file).read_text().strip() if a.hint_file else None
    prior = [Path(f).read_text().strip() for f in (a.prior_hint_file or [])] or None
    base = _load_snapshot_record(a.base_snapshot) if a.base_snapshot else None
    if base is not None:
        base = dict(base, record_path=str(Path(a.base_snapshot).resolve()))
    bdir, meta = build(adapter=_adapter(a.agent), trial_dir=Path(a.trial_dir), atif_step=a.step, out=Path(a.out),
                       cut_line=a.cut_line,
                       mode=a.mode, hint=hint, agent_timeout=a.agent_timeout,
                       task_dir=Path(a.task_dir) if a.task_dir else None, run_id=a.run_id, model=a.model,
                       pin_date=not a.no_pin_date, fingerprint_roots=a.fingerprint_root or None, force=a.force,
                       persist_hint=a.persist_hint, prior_hints=prior, base_snapshot=base)
    print(bdir)
    print(json.dumps({k: meta[k] for k in ("run_id", "task", "resume_atif_step", "cut_line", "agent_budget_s",
                                           "n_replay_actions", "aux_counts", "n_compactions_before_cut",
                                           "orig_request_at_k", "n_hints", "agent_budget_source")},
                     ensure_ascii=False))
    if meta.get("chain"):
        print(json.dumps({"chain": {"parent_run_id": meta["chain"]["parent_run_id"],
                                    "stripped_lines": sum(len(s["removed_lines"]) for s in meta["chain"]["stripped"]),
                                    "materialised": [m["where"] for m in meta["chain"]["materialised"]]}}, ensure_ascii=False))
    if meta.get("base_snapshot"):
        print(json.dumps({"base_snapshot": meta["base_snapshot"]}, ensure_ascii=False))
    if not meta.get("orig_request_at_k"):
        print(f"warning: original request at the cut not located ({meta.get('orig_request_locate_error')}); "
              "the first-request parity gate cannot run for this bundle", file=sys.stderr)
    return 0


def cmd_build_template(a):
    from .sandbox.templates import build_template

    rec = build_template(task_dir=Path(a.task_dir), state_dir=Path(a.state_dir), image=a.image, cpu=a.cpu,
                         memory_mb=a.memory_mb, disk_mib=a.disk_mib)
    print(json.dumps({k: rec.get(k) for k in ("name", "template_id", "status", "seconds", "reason")}))
    return 0 if "ready" in str(rec.get("status")) else 1


def cmd_prepare_snapshot(a):
    from .sandbox import source

    if a.continue_state:
        rec = source.finish_prepared_source(Path(a.continue_state))
    else:
        missing = [n for n in ("task_dir", "bundle", "template", "tag") if not getattr(a, n)]
        if missing:
            raise SystemExit("missing: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        template = a.template
        if template.endswith(".json") and Path(template).is_file():
            t = json.loads(Path(template).read_text())
            if "ready" not in str(t.get("status")):
                raise SystemExit(f"template not ready: {t.get('status')}")
            template = t["template_id"]
        bmeta = json.loads((Path(a.bundle) / "bundle.json").read_text())
        base = (bmeta.get("base_snapshot") or {}).get("snapshot_id")
        if base and a.full_replay:
            raise SystemExit("this bundle's plan.json was trimmed to the post-snapshot tail: rebuild the bundle "
                             "without --base-snapshot for a full replay")
        rec = source.prepare_source(adapter=_adapter(a.agent), task_dir=Path(a.task_dir), bundle=Path(a.bundle),
                                    template=template, tag=a.tag, state_dir=Path(a.state_dir), version=a.version,
                                    exec_cap=a.exec_cap, expected_fingerprint=a.expected_fingerprint,
                                    retain_on_gate_failure=a.retain_on_gate_failure, base_snapshot=base)
    print(json.dumps({k: rec.get(k) for k in ("snapshot_id", "fingerprint_source", "replay_gate_passed",
                                              "base_snapshot")}))
    return 0 if rec.get("snapshot_id") else 1


def _kv(items):
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--agent-env expects KEY=VALUE, got {item!r}")
        if re.search(r"(KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH)", k, re.I):
            raise SystemExit(f"--agent-env {k}: credentials must not go into the job config")
        out[k] = v
    return out


def cmd_launch(a):
    from .launch import launch_one

    bundle = Path(a.bundle)
    meta = json.loads((bundle / "bundle.json").read_text())
    adapter = _adapter(meta.get("agent") or a.agent)
    job = a.job_name or f"asr-{meta['run_id']}{'-fake' if a.fake else ''}"
    if a.dry_run:
        from .launch import build_job_config, validate_job_config
        cfg = build_job_config(adapter=adapter, bundle=bundle, task_dir=Path(a.task_dir), job_name=job,
                              jobs_dir=Path(a.jobs_dir),
                              snapshot_state=Path(a.snapshot_state) if a.snapshot_state else None,
                              snapshot_id=a.snapshot_id, fake=a.fake, verifier=not a.no_verifier, model=a.model,
                              upstream_url=a.upstream_url, extra_env=_kv(a.agent_env),
                              original_trial_dir=a.original_trial_dir or meta.get("trial_dir"),
                              passthrough_args=shlex.split(a.passthrough_args) if a.passthrough_args else None,
                              parity_gate=a.parity_gate, max_model_requests=a.max_model_requests,
                              freeze_first_request=a.freeze_first_request)
        validate_job_config(cfg)
        runs = Path(a.runs_dir) / job
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / "job.json"
        path.write_text(json.dumps(cfg, indent=1) + "\n")
        print(path)
        print(json.dumps(cfg, indent=1))
        return 0
    return launch_one(adapter=adapter, bundle=bundle, task_dir=Path(a.task_dir), job_name=job,
                      jobs_dir=Path(a.jobs_dir), runs_dir=Path(a.runs_dir),
                      snapshot_state=Path(a.snapshot_state) if a.snapshot_state else None,
                      snapshot_id=a.snapshot_id, fake=a.fake, verifier=not a.no_verifier, model=a.model,
                      upstream_url=a.upstream_url, extra_env=_kv(a.agent_env),
                      original_trial_dir=a.original_trial_dir or meta.get("trial_dir"),
                      passthrough_args=shlex.split(a.passthrough_args) if a.passthrough_args else None,
                      parity_gate=a.parity_gate, max_model_requests=a.max_model_requests,
                      freeze_first_request=a.freeze_first_request,
                      midrun_snapshot_every=a.midrun_snapshot_every, midrun_snapshot_poll=a.midrun_snapshot_poll)


def cmd_snapshot_midrun(a):
    """Attach the mid-run snapshotter to a job that is already running (no-op if it has its own)."""
    from .midrun import MidrunSnapshotter

    job_dir = Path(a.jobs_dir).resolve() / a.job
    run_dir = Path(a.runs_dir).resolve() / a.job
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = MidrunSnapshotter(job_dir, record_path=run_dir / "midrun-snapshots.jsonl", every_s=a.every_s,
                             poll_s=a.poll_s)
    if a.once:
        n = snap.tick()
        print(json.dumps({"snapshots": n, "errors": snap.errors[-5:]}, ensure_ascii=False))
        return 0
    snap.run()
    return 0


def cmd_batch(a):
    from .batch import Batch, print_status

    work = Path(a.work_dir).resolve()
    b = Batch(name=a.name, work_dir=work, manifest=Path(a.manifest),
              tasks_dir=Path(a.tasks_dir) if a.tasks_dir else work / "tasks",
              jobs_dir=Path(a.jobs_dir) if a.jobs_dir else work / "jobs",
              runs_dir=Path(a.runs_dir) if a.runs_dir else work / "runs",
              state_dir=Path(a.state_dir) if a.state_dir else work / "state",
              agent_env=_kv(a.agent_env),
              passthrough_args=shlex.split(a.passthrough_args) if a.passthrough_args else None,
              model=a.model, upstream_url=a.upstream_url, hinted_reps=a.hinted_reps,
              control_reps=a.control_reps, midrun_snapshot_every=a.midrun_snapshot_every,
              cut_window=a.cut_window, max_cut_candidates=a.max_cut_candidates, retry_failed=a.retry_failed,
              prep_concurrency=a.prep_concurrency,
              launch_concurrency=a.launch_concurrency, only=a.only.split(",") if a.only else None,
              agent=a.agent, verifier=not a.no_verifier)
    if a.mode == "status":
        rows = b.status()
        if a.json:
            print(json.dumps(rows, indent=1, ensure_ascii=False))
        else:
            print_status(rows)
            done = sum(1 for r in rows if r["state"] == "done")
            running = sum(1 for r in rows if r["state"] == "running")
            print(f"\n{len(rows)} runs: {done} done, {running} running, "
                  f"{len({r['task'] for r in rows})} tasks")
        return 0
    if not a.manifest:
        raise SystemExit("--manifest is required for run/prep/launch")
    if a.mode == "prep":
        res = b.run(do_launch=False)
    elif a.mode == "launch":
        res = b.run(do_launch=True)
    else:
        res = b.run(do_launch=True)
    print(json.dumps({"prepared": len(res["prepared"]), "failed": res["failed"],
                      "launched": len(res["launched"]), "launch_failed": res["launch_failed"]},
                     ensure_ascii=False, indent=1))
    return 0 if not res["failed"] and not res["launch_failed"] else 1


def cmd_equiv(a):
    from . import equiv
    return equiv.main(a)


def cmd_screen(a):
    from . import screen
    reps = screen.screen_trial(_adapter(a.agent), Path(a.trial_dir), cut_lines=a.cut_line, steps=a.step,
                               agent_timeout_s=a.agent_timeout,
                               task_dir=Path(a.task_dir) if a.task_dir else None)
    print(screen.main_json(reps, a.full))
    if a.out:
        Path(a.out).write_text(json.dumps(reps, ensure_ascii=False, indent=1))
    return 2 if any(r["verdict"] == "block" for r in reps) else 0


def cmd_parity(a):
    from . import parity
    r = parity.check_files(a.bundle, a.request, shlex.split(a.passthrough_args) if a.passthrough_args else None)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r.get("ok") else 1


def cmd_selfcheck(a):
    """Offline: simulate the agent's first resumed request from the bundle, rewrite it, check equivalence."""
    from . import equiv
    from .proxy.resume_proxy import rewrite

    bundle = Path(a.bundle)
    meta = json.loads((bundle / "bundle.json").read_text())
    adapter = _adapter(meta.get("agent") or "claude-code")
    simulate = getattr(adapter, "simulate_resume_request", None)
    if simulate is None:
        raise SystemExit(f"adapter {adapter.name} has no simulate_resume_request")
    spec = json.loads((bundle / "proxy_spec.json").read_text())
    with open(bundle / "session_trunc.jsonl", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    raw = simulate(lines, spec["sentinel"])
    new, actions = rewrite(json.loads(json.dumps(raw)), spec)
    out = Path(a.out) if a.out else bundle / "selfcheck"
    out.mkdir(parents=True, exist_ok=True)
    (out / "first_request_raw.json").write_text(json.dumps(raw, ensure_ascii=False))
    (out / "first_request_rewritten.json").write_text(json.dumps(new, ensure_ascii=False))
    print(json.dumps({"rewrite_actions": actions}))
    r = equiv.check(bundle / "session_trunc.jsonl", meta["cut_line"], out / "first_request_rewritten.json",
                    hint=meta.get("hint") if meta.get("mode") == "hint" else None, sentinel=spec["sentinel"])
    (out / "equivalence_report.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps({k: r[k] for k in ("equivalent", "n_ref_blocks", "n_matched_in_order", "n_missing",
                                        "extra_by_class", "sentinel_left", "hint_blocks")}))
    return 0 if r["equivalent"] else 1


def cmd_sandboxes(a):
    from .sandbox import manage

    if a.kill_id:
        print(manage.kill(a.kill_id, force=a.force))
        return 0
    items = manage.list_own()
    for s in items:
        md = s.metadata or {}
        print(json.dumps({"sandbox_id": s.sandbox_id, "role": md.get("role"), "session_id": md.get("session_id"),
                          "run": md.get("asr_run"), "snapshot_id": md.get("snapshot_id"),
                          "started_at": str(s.started_at)}))
        if a.peek:
            try:
                print(manage.peek(s.sandbox_id))
            except Exception as e:  # noqa: BLE001
                print("peek failed", repr(e)[:300])
    if not items:
        print(f"no sandboxes with owner={config.owner_tag()}")
    return 0


def cmd_delete_snapshot(a):
    from .sandbox import manage
    print(manage.delete_snapshot(a.snapshot_id))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="asr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default=".env", help="load E2B_*/ASR_*/FC_SANDBOX_*/REGISTRY_* from this file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-bundle", help="trial dir + step k -> resume bundle")
    p.add_argument("--trial-dir", required=True, help="Harbor trial dir (contains agent/sessions/...)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--step", type=int, help="ATIF step_id of the first step to regenerate")
    g.add_argument("--cut-line", type=int, help="0-based session line where the regenerated agent step starts "
                   "(= number of prefix lines kept); may be after compaction boundaries")
    p.add_argument("--out", default="bundles")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--mode", choices=["same", "hint"], default="same")
    p.add_argument("--hint-file")
    p.add_argument("--persist-hint", action="store_true",
                   help="mode=hint: keep every hint the run carries after the agent compacts its context "
                   "(each re-injected once on the summary)")
    p.add_argument("--prior-hint-file", action="append", metavar="FILE",
                   help="a hint this resumed run already carries (repeatable, oldest first); must match the run's own "
                   "proxy spec. Default: derive the chain from the run's agent/resume/proxy_spec.json")
    p.add_argument("--base-snapshot", metavar="RECORD",
                   help="mid-run snapshot record (a line of runs/<job>/midrun-snapshots.jsonl, or a .json file with "
                   "one): replay only the tail after its session line instead of the whole prefix")
    p.add_argument("--agent-timeout", type=float, help="task agent timeout in s (default: from task.toml)")
    p.add_argument("--task-dir", help="Harbor task dir (for task.toml)")
    p.add_argument("--run-id")
    p.add_argument("--model", help="model name the resumed agent reports (default: the original run's)")
    p.add_argument("--fingerprint-root", action="append", help="absolute task root to fingerprint (repeatable; default /app)")
    p.add_argument("--no-pin-date", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing bundle dir")
    p.set_defaults(fn=cmd_build_bundle)

    p = sub.add_parser("build-template", help="register a snapshot-capable template from the task image")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--image", help="override task.toml [environment] docker_image")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--memory-mb", type=int, default=4096)
    p.add_argument("--disk-mib", type=int, default=15360)
    p.set_defaults(fn=cmd_build_template)

    p = sub.add_parser("prepare-snapshot", help="replay steps < k in a source sandbox and snapshot it")
    p.add_argument("--task-dir")
    p.add_argument("--bundle")
    p.add_argument("--template", help="template id/name, or the state/template-<task>.json written by build-template")
    p.add_argument("--tag", help="unique name for this source attempt")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--version", help="agent version (default: adapter's pinned version)")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--exec-cap", type=int, default=1800, help="max seconds per replayed command")
    p.add_argument("--expected-fingerprint")
    p.add_argument("--retain-on-gate-failure", action="store_true")
    p.add_argument("--full-replay", action="store_true",
                   help="ignore the bundle's base_snapshot and replay the whole prefix from the task template")
    p.add_argument("--continue", dest="continue_state", help="finish a retained source from its state json")
    p.set_defaults(fn=cmd_prepare_snapshot)

    p = sub.add_parser("launch", help="write the Harbor job config and run it")
    p.add_argument("--bundle", required=True)
    p.add_argument("--task-dir", "--tasks-dir", dest="task_dir", required=True, help="Harbor task dir of the trial")
    p.add_argument("--snapshot-state", help="state/snapshot-<tag>.json from prepare-snapshot (enables all gates)")
    p.add_argument("--snapshot-id")
    p.add_argument("--job-name")
    p.add_argument("--jobs-dir", default="jobs")
    p.add_argument("--runs-dir", default="runs", help="where job.json is written")
    p.add_argument("--midrun-snapshot-every", type=float, default=1200.0, metavar="SECONDS",
                   help="snapshot the live clone every N seconds and after each compaction (0 disables); records go "
                   "to <runs-dir>/<job>/midrun-snapshots.jsonl")
    p.add_argument("--midrun-snapshot-poll", type=float, default=60.0, metavar="SECONDS")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--model")
    p.add_argument("--upstream-url", help="default: ASR_UPSTREAM_URL")
    p.add_argument("--original-trial-dir", help="copy non-secret agent env/kwargs from <trial>/config.json")
    p.add_argument("--agent-env", action="append", metavar="KEY=VALUE", help="extra non-secret agent env (repeatable)")
    p.add_argument("--passthrough-args", help="extra in-sandbox passthrough proxy flags, e.g. "
                   "'--drop-fields metadata --strip-is-error' (whitelisted)")
    p.add_argument("--fake", action="store_true", help="resume proxy answers end_turn; no model call, no key")
    p.add_argument("--no-verifier", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write and print job.json, do not start Harbor")
    p.add_argument("--parity-gate", choices=["auto", "require", "off"], default="auto",
                   help="fail-closed first-request parity vs the original run's request at the cut (default auto: "
                   "on whenever the bundle located that request)")
    p.add_argument("--freeze-first-request", metavar="FIRST_REQUEST_JSON",
                   help="first_request_rewritten.json of a --fake run of this bundle that passed the parity gate: the "
                   "real first request must equal it in every field except metadata (required with --parity-gate "
                   "require for real runs)")
    p.add_argument("--max-model-requests", type=int,
                   help="stop the continuation after N model requests, sub-agent requests included (the proxy refuses request N+1)")
    p.set_defaults(fn=cmd_launch)

    from .equiv import build_parser as equiv_parser
    p = sub.add_parser("snapshot-midrun", help="snapshot a running job's live clones (attach to a running launch)")
    p.add_argument("--job", required=True, help="Harbor job name (its trials live in <jobs-dir>/<job>/)")
    p.add_argument("--jobs-dir", default="jobs")
    p.add_argument("--runs-dir", default="runs", help="where midrun-snapshots.jsonl is appended")
    p.add_argument("--every-s", type=float, default=1200.0)
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.set_defaults(fn=cmd_snapshot_midrun)

    p = sub.add_parser("batch", help="manifest-driven batch: screen -> bundles -> template -> snapshot -> fake -> "
                                     "real launches (idempotent, resumable)")
    p.add_argument("mode", choices=["run", "prep", "launch", "status"],
                   help="run = prepare and launch, prep = stop before the real launches, launch = launch what is "
                        "prepared, status = table of every run")
    p.add_argument("--name", required=True, help="batch name (state lives in <work-dir>/batch/<name>/)")
    p.add_argument("--manifest", help="CSV: task,cut_line,hint_path,... (default: round1/manifest.csv columns)")
    p.add_argument("--work-dir", default=".", help="base for jobs/runs/state/tasks and batch/<name>/")
    p.add_argument("--tasks-dir", help="Harbor task dirs (default <work-dir>/tasks)")
    p.add_argument("--jobs-dir", help="Harbor jobs dir (default <work-dir>/jobs)")
    p.add_argument("--runs-dir", help="job.json + midrun-snapshots (default <work-dir>/runs)")
    p.add_argument("--state-dir", help="template/snapshot state (default <work-dir>/state)")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--model")
    p.add_argument("--upstream-url", help="default: ASR_UPSTREAM_URL")
    p.add_argument("--agent-env", action="append", metavar="KEY=VALUE")
    p.add_argument("--passthrough-args", help="in-sandbox passthrough proxy flags for every run")
    p.add_argument("--hinted-reps", type=int, default=2)
    p.add_argument("--control-reps", type=int, default=1)
    p.add_argument("--midrun-snapshot-every", type=float, default=1200.0,
                   help="mid-run snapshots for the hinted runs (0 disables; controls never snapshot)")
    p.add_argument("--cut-window", type=int, default=200,
                   help="when a cut blocks the replay screen, try agent-step starts up to this many lines away "
                        "(always before the fork)")
    p.add_argument("--max-cut-candidates", type=int, default=16)
    p.add_argument("--retry-failed", choices=["off", "infra", "all"], default="off",
                   help="reset failed units before running: 'infra' retries those whose failure was not a replay-gate "
                        "refusal (e.g. an interrupted preparation), 'all' retries every failed unit")
    p.add_argument("--prep-concurrency", type=int, default=6)
    p.add_argument("--launch-concurrency", type=int, default=20)
    p.add_argument("--only", help="comma-separated task names")
    p.add_argument("--no-verifier", action="store_true")
    p.add_argument("--json", action="store_true", help="status: machine-readable")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("equiv", help="context-equivalence check of a rewritten first request")
    equiv_parser(p)
    p.set_defaults(fn=cmd_equiv)

    p = sub.add_parser("screen", help="offline replayability risks of candidate cuts")
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--cut-line", type=int, action="append", help="0-based session line of a candidate cut (repeat)")
    p.add_argument("--step", type=int, action="append", help="ATIF step id of a candidate cut (repeat)")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--agent-timeout", type=float)
    p.add_argument("--task-dir", help="Harbor task dir (for the agent timeout in task.toml)")
    p.add_argument("--full", action="store_true", help="print every risk, not a per-cut summary")
    p.add_argument("--out", help="write the full reports as JSON")
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("parity", help="first rewritten request vs the original run's request at the cut")
    p.add_argument("bundle")
    p.add_argument("request", help="first_request_rewritten.json captured by the resume proxy")
    p.add_argument("--passthrough-args", help="the original route's passthrough flags, e.g. "
                   "'--drop-fields metadata --strip-is-error'")
    p.set_defaults(fn=cmd_parity)

    p = sub.add_parser("selfcheck", help="offline: simulated resumed request -> rewrite -> equiv")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", help="default: <bundle>/selfcheck")
    p.set_defaults(fn=cmd_selfcheck)

    p = sub.add_parser("sandboxes", help="list own sandboxes (metadata owner=ASR_OWNER_TAG)")
    p.add_argument("--kill-id")
    p.add_argument("--force", action="store_true", help="kill even if not tagged with the owner tag")
    p.add_argument("--peek", action="store_true", help="show proxy log tails inside each sandbox")
    p.set_defaults(fn=cmd_sandboxes)

    p = sub.add_parser("delete-snapshot", help="delete a snapshot by id")
    p.add_argument("snapshot_id")
    p.set_defaults(fn=cmd_delete_snapshot)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    config.load_env_file(a.env_file)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
