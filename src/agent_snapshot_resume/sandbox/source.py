"""Build the step-k environment snapshot (host side; synchronous E2B SDK).

One source sandbox from the task's snapshot-capable template:
  1. task healthcheck command (from task.toml, as Harbor runs it) + chmod 1777 /tmp
  2. the agent adapter's pinned install (sha256-checked) and the in-sandbox proxy runtime (no key, no proxy started)
  3. replay of steps < k with replay/replay_inside.py and the bundle's plan.json (dispatched exactly once)
  4. fail-closed replay gate, auxiliary-file hashes, fingerprint of the task roots
  5. remove replay scaffolding and logs, snapshot, kill the source sandbox
No model call happens and no key ever enters the source sandbox, so the snapshot holds no secret.
State (including the snapshot id and the fingerprint the clones must match) is written to
<state_dir>/snapshot-<tag>.json. If replay completion is not observed in time, or the gate needs review, the source can
be retained (`--retain-on-gate-failure`) and finished later with `finish_prepared_source` (never re-dispatches).
"""
from __future__ import annotations

import hashlib
import json
import shlex
import time
import tomllib
from pathlib import Path
from typing import Any

from ..config import owner_tag
from ..proxy import runtime
from ..replay.fingerprint import DEFAULT_ROOTS, build_fingerprint_command
from ..replay.gate import validate_replay

REPLAY_DIR = "/tmp/.asr-resume"
REPLAY_SCRIPT = Path(__file__).resolve().parent.parent / "replay" / "replay_inside.py"
REPLAY_DEADLINE_S = 10800


def dispatch_replay(sbx, cwd: str, exec_cap: int) -> int:
    """Start the replay exactly once in SDK background mode; completion is signalled by an atomic rc file."""
    command = (f'cd {shlex.quote(cwd)} && timeout -s TERM -k 10s {REPLAY_DEADLINE_S}s '
               f'python3 {REPLAY_DIR}/replay_inside.py {REPLAY_DIR}/plan.json '
               f'{REPLAY_DIR}/replay_report.json --exec-cap {int(exec_cap)} '
               f'> {REPLAY_DIR}/replay_stdout.log 2>&1; rc=$?; '
               f'printf "%s\\n" "$rc" > {REPLAY_DIR}/rc.tmp; '
               f'mv {REPLAY_DIR}/rc.tmp {REPLAY_DIR}/rc')
    # One dispatch only. SDK-background detaches the process; a shell '&' did not.
    return sbx.commands.run(command, user='root', background=True, timeout=0).pid


def retain_for_audit(requested: bool, plan: dict, report: dict, rc) -> bool:
    """Keep a source sandbox only if asked AND the replay completed every action (so it is worth reviewing)."""
    return bool(requested and str(rc).strip() == '0' and
                len(report.get('results', [])) == len(plan['actions']))


def fingerprint_command_for(meta: dict[str, Any]) -> str:
    roots = meta.get("fingerprint_roots") or list(DEFAULT_ROOTS)
    return build_fingerprint_command(roots, meta.get("fingerprint_optional_roots", []))


def task_healthcheck(task_dir: Path) -> str | None:
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    return ((cfg.get("environment") or {}).get("healthcheck") or {}).get("command")


def _sandbox_cls():
    from e2b import Sandbox
    return Sandbox


def _install_proxy_runtime(sbx, run) -> None:
    files = runtime.proxy_file_bytes()
    run(sbx, "rm -rf /tmp/.asr-proxy-src && mkdir -p /tmp/.asr-proxy-src")
    for name, data in files.items():
        sbx.files.write(f"/tmp/.asr-proxy-src/{name}", data)
    sbx.files.write("/tmp/.asr-proxy-src/SHA256SUMS", runtime.sha256sums(files).encode())
    import certifi
    sbx.files.write("/tmp/.asr-cacert.pem", Path(certifi.where()).read_bytes())
    run(sbx, runtime.proxy_place_command(), timeout=1200)
    run(sbx, runtime.CA_LINK)


def _collect_gate_evidence(sbx, rec, rep, save):
    rec["reviewed_replay_exceptions"] = [{"call_id": x["call_id"], "reason": x["reviewed_exception"]}
                                         for x in rep["results"] if x.get("reviewed_exception")]
    rec["auxiliary_hashes"] = {}
    for item in rep["results"]:
        if item.get("background_output_file"):
            fp = item["background_output_file"]
            h = hashlib.sha256(bytes(sbx.files.read(fp, format="bytes"))).hexdigest()
            if h != item["background_output_sha256"]:
                raise RuntimeError("Restored background output hash mismatch")
            rec["auxiliary_hashes"][fp] = h
    rec["replay_gate_passed"] = True
    save()


def _snapshot_and_kill(sbx, rec, step, run, fingerprint_cmd, expected_fingerprint=None):
    t = time.time()
    run(sbx, f"rm -rf {REPLAY_DIR} /logs/agent /logs/verifier /logs/artifacts; sync")
    rec["fingerprint_command"] = fingerprint_cmd
    fp = run(sbx, fingerprint_cmd, timeout=600).stdout.strip()
    if rec.get("pending_audit_fingerprint") and fp != rec["pending_audit_fingerprint"]:
        raise RuntimeError("Task files changed during completion audit")
    rec["fingerprint_source"] = fp
    if expected_fingerprint and fp.splitlines()[1] != expected_fingerprint:
        raise RuntimeError("Source task-file fingerprint differs from expected reference")
    rec["processes_source"] = run(sbx, "ps -eo pid,etime,args --sort=pid | grep -v -E 'ps -eo|grep' | tail -20").stdout.strip()
    step("fingerprint", t)
    t = time.time()
    rec["snapshot_dispatch_started"] = True
    try:
        snap = sbx.create_snapshot(request_timeout=300)
        snapshot_id, names = snap.snapshot_id, list(snap.names or [])
    except Exception as exc:  # noqa: BLE001
        # The create call can lose its response (e.g. an HTTP/2 stream reset) after the service accepted it.
        # Never dispatch a second snapshot: look up the one this sandbox produced instead.
        rec["snapshot_create_error"] = repr(exc)[:300]
        found = recover_snapshot_receipt(sbx)
        if found is None:
            raise
        snapshot_id, names = found
        rec["snapshot_receipt_recovered"] = True
    rec["snapshot_id"] = snapshot_id
    rec["snapshot_names"] = names
    rec["snapshot_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    step("snapshot", t, snapshot_id=snapshot_id)


def recover_snapshot_receipt(sbx, attempts: int = 30, delay_s: float = 10.0):
    """Return (snapshot_id, names) of the single snapshot of this sandbox, or None. Fails closed on ambiguity."""
    for _ in range(attempts):
        try:
            items = sbx.list_snapshots().next_items()
        except Exception:  # noqa: BLE001
            items = None
        if items is not None and len(items) > 1:
            raise RuntimeError(f"{len(items)} snapshots exist for this source sandbox; resolve by hand")
        if items:
            return items[0].snapshot_id, list(items[0].names or [])
        time.sleep(delay_s)
    return None


def kill_source(sbx, attempts: int = 30, delay_s: float = 10.0) -> None:
    """Kill the source sandbox, waiting while the service still reports it as snapshotting (HTTP 409)."""
    for i in range(attempts):
        try:
            sbx.kill()
            return
        except Exception as exc:  # noqa: BLE001
            if "409" not in str(exc) or i == attempts - 1:
                raise
            time.sleep(delay_s)


def prepare_source(*, adapter, task_dir: Path, bundle: Path, template: str, tag: str, state_dir: Path,
                   version: str | None = None, exec_cap: int = 1800, expected_fingerprint: str | None = None,
                   retain_on_gate_failure: bool = False, sandbox_timeout_s: int = 4 * 3600) -> dict[str, Any]:
    Sandbox = _sandbox_cls()
    state_dir.mkdir(parents=True, exist_ok=True)
    out = state_dir / f"snapshot-{tag}.json"
    if out.exists():
        raise RuntimeError(f"{out} exists; refusing a duplicate attempt (use a new --tag)")
    meta = json.loads((bundle / "bundle.json").read_text())
    adapter.sanity_check_bundle(bundle, meta)
    fingerprint_cmd = fingerprint_command_for(meta)
    healthcheck = task_healthcheck(task_dir)
    owner = owner_tag()
    from ..bundle import snapshot_cut_identity
    # Default to the original run's agent version (recorded in the bundle), not the adapter's default.
    version = version or meta.get("agent_version_orig") or adapter.version
    rec: dict[str, Any] = {"task_dir": str(task_dir), "tag": tag, "template": template, "bundle": str(bundle.resolve()),
                           "run_id": meta["run_id"], "resume_atif_step": meta["resume_atif_step"],
                           "cut_identity": snapshot_cut_identity(bundle),
                           "agent": adapter.name, "agent_version": version, "owner": owner,
                           "steps": []}

    def save():
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")

    def step(name, t0, **kw):
        rec["steps"].append({"step": name, "s": round(time.time() - t0, 1), **kw})
        save()
        print(json.dumps(rec["steps"][-1], ensure_ascii=False), flush=True)

    def run(sbx, cmd, timeout=900, envs=None, cwd=None):
        return sbx.commands.run(cmd, user="root", timeout=timeout, envs=envs, cwd=cwd)

    t = time.time()
    sbx = Sandbox.create(template=template, timeout=sandbox_timeout_s,
                         metadata={"owner": owner, "role": "source", "asr_run": tag})
    rec["source_sandbox_id"] = sbx.sandbox_id
    step("create_source", t, sandbox_id=sbx.sandbox_id)
    try:
        t = time.time()
        run(sbx, (healthcheck + " && " if healthcheck else "") + "chmod 1777 /tmp")
        r = run(sbx, "uname -a; nproc; free -m | head -2; df -h / | tail -1; cat /etc/os-release | head -2")
        step("healthcheck", t, system=r.stdout[-600:])

        t = time.time()
        for name, cmd, timeout in adapter.install_commands(version):
            r = run(sbx, cmd, timeout=timeout, envs={"DEBIAN_FRONTEND": "noninteractive"})
        step("install_agent", t, version=(r.stdout or "").strip()[-200:])

        t = time.time()
        _install_proxy_runtime(sbx, run)
        step("install_proxy_runtime", t)

        t = time.time()
        run(sbx, f"mkdir -p {REPLAY_DIR}")
        sbx.files.write(f"{REPLAY_DIR}/plan.json", (bundle / "plan.json").read_bytes())
        sbx.files.write(f"{REPLAY_DIR}/replay_inside.py", REPLAY_SCRIPT.read_bytes())
        cwd = meta.get("cwd") or "/app"
        rec["replay_pid"] = dispatch_replay(sbx, cwd, exec_cap)
        step("replay_dispatched_once", t, pid=rec["replay_pid"])
        while True:
            time.sleep(15)
            r = run(sbx, f"cat {REPLAY_DIR}/rc 2>/dev/null || true", timeout=60)
            if r.stdout.strip():
                break
            if time.time() - t > REPLAY_DEADLINE_S + 120:
                rec["source_retained_for_audit"] = True
                save()
                raise TimeoutError("replay completion not observed; finish later with --continue (never redispatch)")
        rc = r.stdout.strip()
        for name in ("replay_report.json", "replay_stdout.log"):
            (state_dir / f"{tag}-{name}").write_bytes(bytes(sbx.files.read(f"{REPLAY_DIR}/{name}", format="bytes")))
        rep = json.loads((state_dir / f"{tag}-replay_report.json").read_text())
        step("replay", t, rc=rc, summary={k: v for k, v in rep.items() if k != "results"})
        plan = json.loads((bundle / "plan.json").read_text())
        try:
            validate_replay(plan, rep, rc, meta.get("replay_reviewed_exceptions", {}))
        except RuntimeError as exc:
            if retain_for_audit(retain_on_gate_failure, plan, rep, rc):
                rec["replay_gate_error"] = str(exc)
                rec["replay_gate_passed"] = False
                rec["fingerprint_command"] = fingerprint_cmd
                rec["pending_audit_fingerprint"] = run(sbx, fingerprint_cmd, timeout=600).stdout.strip()
                rec["pending_plan_sha256"] = hashlib.sha256((bundle / "plan.json").read_bytes()).hexdigest()
                rec["pending_report_sha256"] = hashlib.sha256(
                    (state_dir / f"{tag}-replay_report.json").read_bytes()).hexdigest()
                rec["source_retained_for_audit"] = True
                save()
            raise
        (state_dir / f"{tag}-replay_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        _collect_gate_evidence(sbx, rec, rep, save)
        _snapshot_and_kill(sbx, rec, step, run, fingerprint_cmd, expected_fingerprint)
    finally:
        t = time.time()
        if rec.get("source_retained_for_audit"):
            step("retained_source_requires_gate_review_or_cleanup", t)
        else:
            try:
                kill_source(sbx)
                step("kill_source", t)
            except Exception as e:  # noqa: BLE001
                step("kill_source_failed", t, error=repr(e)[:300])
    save()
    return rec


def finish_prepared_source(state_path: Path, *, observe_timeout_s: int = 10800) -> dict[str, Any]:
    """Finish an already-dispatched replay on a retained source sandbox. Never dispatches commands again.

    Re-validates the replay with the bundle's (possibly updated) reviewed exceptions, refuses if the task fingerprint
    or replay evidence changed since it was recorded, then snapshots and kills the source.
    """
    Sandbox = _sandbox_cls()
    state_path = Path(state_path)
    state_dir = state_path.parent
    rec = json.loads(state_path.read_text())
    if rec.get("snapshot_id") or rec.get("snapshot_dispatch_started"):
        raise RuntimeError("Snapshot already dispatched; inspect the existing snapshot instead of creating another")
    bundle = Path(rec["bundle"])
    meta = json.loads((bundle / "bundle.json").read_text())

    def save():
        state_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")

    def step(name, t0, **kw):
        rec.setdefault("steps", []).append({"step": name, "s": round(time.time() - t0, 1), **kw})
        save()
        print(json.dumps(rec["steps"][-1], ensure_ascii=False), flush=True)

    def run(sbx, cmd, timeout=900):
        return sbx.commands.run(cmd, user="root", timeout=timeout)

    sb = Sandbox.connect(rec["source_sandbox_id"])
    info = sb.get_info()
    if info.metadata.get("owner") != rec.get("owner", owner_tag()) or info.metadata.get("role") != "source" \
            or info.metadata.get("asr_run") != rec["tag"]:
        raise RuntimeError("source sandbox metadata does not match the state file")
    if bytes(sb.files.read(f"{REPLAY_DIR}/plan.json", format="bytes")) != (bundle / "plan.json").read_bytes():
        raise RuntimeError("plan in the source sandbox differs from the bundle")
    if bytes(sb.files.read(f"{REPLAY_DIR}/replay_inside.py", format="bytes")) != REPLAY_SCRIPT.read_bytes():
        raise RuntimeError("replayer in the source sandbox differs from this package")
    rec["completion_observer_handoff"] = {"replay_pid": rec.get("replay_pid"), "redispatched": False}
    rec["source_retained_for_audit"] = True
    save()
    t = time.time()
    deadline = time.monotonic() + observe_timeout_s
    while True:
        rc = run(sb, f"cat {REPLAY_DIR}/rc 2>/dev/null || true", timeout=30).stdout.strip()
        if rc:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Remote replay completion not observed; do not redispatch")
        time.sleep(15)
    for name in ("replay_report.json", "replay_stdout.log"):
        (state_dir / f"{rec['tag']}-{name}").write_bytes(bytes(sb.files.read(f"{REPLAY_DIR}/{name}", format="bytes")))
    plan = json.loads((bundle / "plan.json").read_text())
    rp = state_dir / f"{rec['tag']}-replay_report.json"
    report = json.loads(rp.read_text())
    fingerprint_cmd = fingerprint_command_for(meta)
    fingerprint = run(sb, fingerprint_cmd, timeout=600).stdout.strip()
    evidence = [("pending_audit_fingerprint", fingerprint),
                ("pending_plan_sha256", hashlib.sha256((bundle / "plan.json").read_bytes()).hexdigest()),
                ("pending_report_sha256", hashlib.sha256(rp.read_bytes()).hexdigest())]
    for key, value in evidence:
        if key in rec and rec[key] != value:
            raise RuntimeError("Previously recorded task fingerprint changed or replay evidence changed: " + key)
        rec[key] = value
    save()
    try:
        validate_replay(plan, report, rc, meta.get("replay_reviewed_exceptions", {}))
    except RuntimeError as exc:
        rec["replay_gate_error"] = str(exc)
        rec["replay_gate_passed"] = False
        save()
        raise
    rec.pop("replay_gate_error", None)
    # Reviewed exceptions may have been added to the bundle after prepare; the snapshot is gated with these.
    from ..bundle import snapshot_cut_identity
    rec["cut_identity"] = snapshot_cut_identity(bundle)
    _collect_gate_evidence(sb, rec, report, save)
    _snapshot_and_kill(sb, rec, step, run, fingerprint_cmd)
    t = time.time()
    kill_source(sb)
    rec["source_retained_for_audit"] = False
    step("kill_source", t)
    return rec
