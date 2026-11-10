"""`asr` command line. Run `asr <command> --help` for details.

  build-bundle      Harbor trial dir + step k -> resume bundle (offline)
  build-template    register a snapshot-capable sandbox template from a task image
  prepare-snapshot  source sandbox: install agent, replay steps < k, gate, fingerprint, snapshot
  launch            write the Harbor job config (and run Harbor unless --dry-run)
  equiv             check a rewritten first request against the original session prefix (offline)
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


def cmd_build_bundle(a):
    from .bundle import build

    hint = Path(a.hint_file).read_text().strip() if a.hint_file else None
    bdir, meta = build(adapter=_adapter(a.agent), trial_dir=Path(a.trial_dir), atif_step=a.step, out=Path(a.out),
                       mode=a.mode, hint=hint, agent_timeout=a.agent_timeout,
                       task_dir=Path(a.task_dir) if a.task_dir else None, run_id=a.run_id, model=a.model,
                       pin_date=not a.no_pin_date, fingerprint_roots=a.fingerprint_root or None, force=a.force)
    print(bdir)
    print(json.dumps({k: meta[k] for k in ("run_id", "task", "cut_line", "agent_budget_s", "n_replay_actions",
                                           "aux_counts")}, ensure_ascii=False))
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
        rec = source.prepare_source(adapter=_adapter(a.agent), task_dir=Path(a.task_dir), bundle=Path(a.bundle),
                                    template=template, tag=a.tag, state_dir=Path(a.state_dir), version=a.version,
                                    exec_cap=a.exec_cap, expected_fingerprint=a.expected_fingerprint,
                                    retain_on_gate_failure=a.retain_on_gate_failure)
    print(json.dumps({k: rec.get(k) for k in ("snapshot_id", "fingerprint_source", "replay_gate_passed")}))
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
    from .launch import build_job_config, run_harbor, validate_job_config

    bundle = Path(a.bundle)
    meta = json.loads((bundle / "bundle.json").read_text())
    adapter = _adapter(meta.get("agent") or a.agent)
    job = a.job_name or f"asr-{meta['run_id']}{'-fake' if a.fake else ''}"
    cfg = build_job_config(adapter=adapter, bundle=bundle, task_dir=Path(a.task_dir), job_name=job,
                           jobs_dir=Path(a.jobs_dir), snapshot_state=Path(a.snapshot_state) if a.snapshot_state else None,
                           snapshot_id=a.snapshot_id, fake=a.fake, verifier=not a.no_verifier, model=a.model,
                           upstream_url=a.upstream_url, extra_env=_kv(a.agent_env),
                           original_trial_dir=a.original_trial_dir or meta.get("trial_dir"),
                           passthrough_args=shlex.split(a.passthrough_args) if a.passthrough_args else None)
    validate_job_config(cfg)
    runs = Path(a.runs_dir) / job
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "job.json"
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    if a.dry_run:
        print(path)
        print(json.dumps(cfg, indent=1))
        return 0
    return run_harbor(path, a.fake)


def cmd_equiv(a):
    from . import equiv
    return equiv.main(a)


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
    p.add_argument("--step", type=int, required=True, help="ATIF step_id of the first step to regenerate")
    p.add_argument("--out", default="bundles")
    p.add_argument("--agent", default="claude-code")
    p.add_argument("--mode", choices=["same", "hint"], default="same")
    p.add_argument("--hint-file")
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
    p.set_defaults(fn=cmd_launch)

    from .equiv import build_parser as equiv_parser
    p = sub.add_parser("equiv", help="context-equivalence check of a rewritten first request")
    equiv_parser(p)
    p.set_defaults(fn=cmd_equiv)

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
