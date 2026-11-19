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

import glob
import hashlib
import json
import os
import shlex
import time
import tomllib
from pathlib import Path
from typing import Any

from ..config import owner_tag
from ..proxy import runtime
from ..replay.fingerprint import DEFAULT_ROOTS, build_fingerprint_command
from ..replay.gate import lenient_report, validate_replay

REPLAY_DIR = "/tmp/.asr-resume"
REPLAY_SCRIPT = Path(__file__).resolve().parent.parent / "replay" / "replay_inside.py"
REPLAY_DEADLINE_S = 10800


POSTCUT_SCRIPT = """import json, os, sys
paths = [p["path"] for p in json.load(open(sys.argv[1]))]
present = {}
for p in paths:
    present[p] = bool(os.path.exists(p) or os.path.lexists(p))
json.dump(present, open(sys.argv[2], "w"), indent=1)
"""


def probe_post_cut(sbx, run, plan: dict, state_dir: Path, tag: str) -> dict:
    """Existing/absent state of the files the agent reads after the cut (lenient gate rule 2).

    The paths come from the bundle (post_cut_paths, built from the original session's post-cut tool calls); the probe
    runs in the replayed environment right after the replay."""
    paths = plan.get("post_cut_paths") or []
    if not paths:
        return {"n": 0, "present": 0, "missing": []}
    sbx.files.write(f"{REPLAY_DIR}/postcut_paths.json", json.dumps(paths).encode())
    sbx.files.write(f"{REPLAY_DIR}/postcut_probe.py", POSTCUT_SCRIPT.encode())
    run(sbx, f"python3 {REPLAY_DIR}/postcut_probe.py {REPLAY_DIR}/postcut_paths.json "
             f"{REPLAY_DIR}/postcut_presence.json")
    raw = json.loads(bytes(sbx.files.read(f"{REPLAY_DIR}/postcut_presence.json", format="bytes")).decode())
    missing = sorted(p for p, ok in raw.items() if not ok)
    (state_dir / f"{tag}-post_cut_paths.json").write_text(json.dumps({"present": raw}, indent=1) + "\n")
    return {"n": len(raw), "present": len(raw) - len(missing), "missing": missing}


def transcript_prefix(meta: dict[str, Any], bundle: Path) -> bytes:
    """The transcript the previous run really had at this cut (its own session file, resume scaffolding included).

    The normalized seed (session_trunc.jsonl) is what Claude Code rebuilds its context from; the parent's file also
    carried its resume scaffolding, and agents do read their own transcript back. Seeding the parent's prefix keeps
    those commands byte-identical at the cut. Falls back to the seed when the parent session is not available.
    """
    seed = (bundle / "session_trunc.jsonl").read_bytes()
    chain_meta = meta.get("chain") or {}
    parent = chain_meta.get("parent_trial_dir")
    if not parent or not meta.get("cut_line"):
        return seed
    removed = [i for a in (chain_meta.get("stripped") or []) for i in a.get("removed_lines") or []]
    cut = int(meta["cut_line"])
    # cut_stripped = cut_orig - #{removed lines < cut_orig}: start from the largest possible answer (all scaffolding
    # before the cut) and descend to the fixed point, so a cut right next to a removed range still maps correctly.
    cut_orig = cut + len(removed)
    while True:
        nxt = cut + sum(1 for i in removed if i < cut_orig)
        if nxt == cut_orig:
            break
        if nxt > cut_orig:  # not a fixed point: never happens for scaffolding before the cut
            raise RuntimeError(f"cannot map the seed cut {cut} back to the parent session")
        cut_orig = nxt
    paths = sorted(glob.glob(os.path.join(parent, "agent", "sessions", "projects", "*", "*.jsonl")))
    if not paths:
        return seed
    with open(paths[0], encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    if cut_orig > len(lines):
        return seed
    return ("\n".join(lines[:cut_orig]) + "\n").encode()


def _write_in_parts(sbx, remote: str, data: bytes, part: int = 8 * 1024 * 1024) -> None:
    """Write a file into the guest in parts: one API write request cannot carry a 40 MB session seed (it times
    out, and the run then starts without a session at all)."""
    if len(data) <= part:
        sbx.files.write(remote, data)
        return
    parts = []
    for i, lo in enumerate(range(0, len(data), part)):
        rp = f"{remote}.part{i}"
        sbx.files.write(rp, data[lo:lo + part])
        parts.append(rp)
    sbx.commands.run("cat " + shlex.join(parts) + " > " + shlex.quote(remote) + " && rm -f " + shlex.join(parts),
                     user="root", timeout=180)
    if not sbx.files.read(remote, format="bytes")[:64]:
        raise RuntimeError(f"part upload of {remote} produced an empty file")


def dispatch_replay(sbx, cwd: str, exec_cap: int, replay_deadline_s: int = REPLAY_DEADLINE_S, pi_user=None) -> int:
    """Start the replay exactly once in SDK background mode; completion is signalled by an atomic rc file.

    The replay runs in a LOGIN shell, as the original agent's shell did (Harbor's task healthcheck installs the
    task's environment through /etc/profile.d/*.sh). A plain `bash -c` inherits only the image's ENV PATH, whose
    first entry is typically the base conda environment, so a task whose image selects another env replays with the
    wrong interpreter (e.g. /opt/conda/bin/python3 without numpy instead of /opt/conda/envs/stereo/bin/python3)."""
    user_arg = "" if pi_user is None else " --pi-user " + shlex.quote(str(pi_user))
    command = (f'cd {shlex.quote(cwd)} && timeout -s TERM -k 10s {int(replay_deadline_s)}s '
               f'python3 {REPLAY_DIR}/replay_inside.py {REPLAY_DIR}/plan.json '
               f'{REPLAY_DIR}/replay_report.json --exec-cap {int(exec_cap)}{user_arg} '
               f'> {REPLAY_DIR}/replay_stdout.log 2>&1; rc=$?; '
               f'printf "%s\\n" "$rc" > {REPLAY_DIR}/rc.tmp; '
               f'mv {REPLAY_DIR}/rc.tmp {REPLAY_DIR}/rc')
    # One dispatch only. SDK-background detaches the process; a shell '&' did not.
    shell = "/bin/bash -c" if pi_user is not None else "bash -lc"
    return sbx.commands.run(f'{shell} {shlex.quote(command)}', user='root', background=True, timeout=0).pid


def retain_for_audit(requested: bool, plan: dict, report: dict, rc) -> bool:
    """Keep a source sandbox only if asked AND the replay completed every action (so it is worth reviewing)."""
    return bool(requested and str(rc).strip() == '0' and
                len(report.get('results', [])) == len(plan['actions']))


def fingerprint_command_for(meta: dict[str, Any]) -> str:
    roots = meta.get("fingerprint_roots") or list(DEFAULT_ROOTS)
    return build_fingerprint_command(roots, meta.get("fingerprint_optional_roots", []))


def native_user_refusal(agent, task_dir, plan, policy):
    if agent not in ("pi", "codex") or policy != "strict" or not plan["actions"]:
        return None
    user = tomllib.loads((task_dir / "task.toml").read_text()).get("agent", {}).get("user")
    if user is not None and str(user) not in ("root", "0"):
        return f"strict {agent} workspace replay cannot reproduce task user {user!r}, group and HOME"
    return None


def check_native_replay(agent, task_dir, bundle, policy):
    if agent not in ("pi", "codex"):
        return
    from ..agents.native import NATIVE_REPLAY_REVISION
    plan = json.loads((bundle / "plan.json").read_bytes())
    issue = native_user_refusal(agent, task_dir, plan, policy)
    if issue:
        raise ValueError(issue)
    if plan.get("native_replay_revision") != NATIVE_REPLAY_REVISION:
        raise ValueError("rebuild the native bundle to check its historical workspace actions")
    if plan.get("replay_refusals"):
        raise ValueError("; ".join(plan["replay_refusals"]))


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
    pi_paths = {r["pi_output_file"] for r in rep["results"] if r.get("pi_output_file")}
    manifest = rep.get("pi_output_manifest", {})
    if pi_paths != set(manifest):
        raise RuntimeError("missing Pi cut output manifest")
    expected = [(r["background_output_file"], r["background_output_sha256"]) for r in rep["results"]
                if r.get("background_output_file")]
    for path, info in manifest.items():
        if info is None:
            quoted = shlex.quote(path)
            sbx.commands.run(f"test ! -e {quoted} && test ! -L {quoted}", user="root")
        else:
            expected.append((path, info["sha256"]))
    for path, digest in expected:
        if hashlib.sha256(bytes(sbx.files.read(path, format="bytes"))).hexdigest() != digest:
            raise RuntimeError("Restored auxiliary file hash mismatch")
        rec["auxiliary_hashes"][path] = digest
    if pi_paths:
        rec["pi_output_manifest"] = manifest
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
                   retain_on_gate_failure: bool = False, sandbox_timeout_s: int = 4 * 3600,
                   base_snapshot: str | None = None, replay_policy: str = "strict",
                   replay_deadline_s: int = REPLAY_DEADLINE_S) -> dict[str, Any]:
    if replay_deadline_s <= 0:
        raise ValueError("replay_deadline_s must be positive")
    check_native_replay(adapter.name, task_dir, bundle, replay_policy)
    sandbox_timeout_s = max(sandbox_timeout_s, replay_deadline_s + 3600)
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
                           "replay_policy": replay_policy, "replay_deadline_s": replay_deadline_s, "steps": []}

    def save():
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")

    def step(name, t0, **kw):
        rec["steps"].append({"step": name, "s": round(time.time() - t0, 1), **kw})
        save()
        print(json.dumps(rec["steps"][-1], ensure_ascii=False), flush=True)

    def run(sbx, cmd, timeout=900, envs=None, cwd=None):
        return sbx.commands.run(cmd, user="root", timeout=timeout, envs=envs, cwd=cwd)

    t = time.time()
    sbx = Sandbox.create(template=base_snapshot or template, timeout=sandbox_timeout_s,
                         metadata={"owner": owner, "role": "source", "asr_run": tag,
                                   **({"base_snapshot": base_snapshot} if base_snapshot else {})})
    rec["source_sandbox_id"] = sbx.sandbox_id
    rec["base_snapshot"] = base_snapshot
    step("create_source", t, sandbox_id=sbx.sandbox_id, base_snapshot=base_snapshot)
    try:
        if base_snapshot:
            # Stop the previous run's processes by pid: a shell `pkill -f <pattern>` would also match the shell that
            # runs it (its own command line contains the pattern) and could abort the cleanup half way.
            t = time.time()
            kill = """
import json, os, subprocess, time
me = os.getpid()
killed = []
# The proxy runtime is started by wrapper scripts (/root/.asr-proxy/run-4013.sh, run-resume.sh) that respawn their
# child when it dies, so the wrappers must be matched too: the leading dot covers both /root/.asr-proxy and
# /opt/asr-proxy without matching this script's own command line.
MARKS = ("/claude", "claude-code", "claude --", ".asr-proxy", "asr_proxy_runtime", "resume_proxy",
         "passthrough_proxy")


def targets():
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = parts
        if int(pid) in (me, os.getppid()) or "/bin/ps" in args or "ps -eo" in args:
            continue
        if "pkill" in args or "asr-base-cleanup" in args:
            continue
        if any(k in args for k in MARKS):
            hits.append(int(pid))
    return hits


# A restored clone carries the previous run's agent and its proxy runtime. The proxy runtime respawns its children
# when they die, so one pass can leave a fresh proxy behind (and it holds the proxy ports the replay needs). The
# restored VM also needs a moment before those processes are visible at all, so scan-kill-scan and only stop after
# two consecutive empty scans.
quiet = 0
for _ in range(12):
    hits = targets()
    if not hits:
        quiet += 1
        if quiet >= 2:
            break
        time.sleep(1.0)
        continue
    quiet = 0
    for pid in hits:
        try:
            os.kill(pid, 9)
            killed.append(pid)
        except OSError:
            pass
    time.sleep(0.5)
print(json.dumps({"killed": killed}))
"""
            cleanup_python = "python3"
            if adapter.name == "pi":
                # Reuse the pinned interpreter already present on a prepared base snapshot.
                cleanup_python = f'$(command -v python3 || echo {runtime.RUNTIME})'
            kill_cmd = "echo " + __import__("base64").b64encode(kill.encode()).decode() + f" | base64 -d | {cleanup_python} -"

            def survivors(tag: str) -> list[str]:
                ps = run(sbx, "ps -eo pid=,ppid=,args=").stdout.splitlines()
                return [l for l in ps if any(k in l for k in ("/claude", ".asr-proxy", "asr_proxy_runtime",
                                                              "resume_proxy", "passthrough_proxy"))
                        and "/bin/ps" not in l and "ps -eo" not in l]

            killed: list[int] = []
            left: list[str] = []
            for attempt in range(3):
                r = run(sbx, kill_cmd, timeout=180)
                killed += json.loads((r.stdout or "{}").strip().splitlines()[-1] or "{}").get("killed", [])
                # A restored clone can show the previous run's processes only after the guest settles, and a proxy
                # wrapper restarts its child when it dies: stop again instead of failing on the first survivor.
                run(sbx, "sleep 3; rm -rf /logs/agent /logs/verifier /logs/artifacts /tmp/.asr-resume", timeout=120)
                left = survivors("cleanup")
                if not left:
                    break
            rec["base_snapshot_killed"] = killed
            if left:
                raise RuntimeError("live clone processes survived the base-snapshot cleanup (3 passes): "
                                   + "\n".join(left[-6:]))
            step("base_snapshot_cleanup", t, base_snapshot=base_snapshot, killed=len(killed))
        t = time.time()
        run(sbx, (healthcheck + " && " if healthcheck else "") + "chmod 1777 /tmp")
        r = run(sbx, "uname -a; nproc; free -m | head -2; df -h / | tail -1; cat /etc/os-release | head -2")
        # Evidence for the replay environment: the login shell the replay uses must resolve the same tools the
        # original agent's shell did (the healthcheck above installs the task env into /etc/profile.d).
        r2_ = run(sbx, "bash -lc 'command -v python3; python3 -c \"import sys; print(sys.executable)\"; "
                      "echo PATH=$PATH'", timeout=180)
        step("healthcheck", t, system=r.stdout[-600:], agent_shell=(r2_.stdout or "").strip().splitlines()[:3])

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
        # The parent run's own transcript was part of its environment: agents do read it back (e.g. to recover a
        # plan after a compaction). Seed it before the tail replay so those commands see the same prefix they saw,
        # at the same path the agent will use for the resumed session.
        slug, sid = meta.get("project_slug"), meta.get("session_id")
        if slug and sid:
            sess_path = (adapter.replay_session_path(meta) if getattr(adapter, "wire", None)
                         else f"/logs/agent/sessions/projects/{slug}/{sid}.jsonl")
            run(sbx, "mkdir -p " + shlex.quote(str(Path(sess_path).parent)))
            _write_in_parts(sbx, sess_path, transcript_prefix(meta, bundle))
        cwd = meta.get("initial_cwd") or meta.get("cwd") or "/app"
        replay_options = {}
        if meta.get("agent") == "pi":
            agent_settings = tomllib.loads((task_dir / "task.toml").read_text()).get("agent", {})
            replay_options["pi_user"] = agent_settings.get("user", "root")
        rec["replay_pid"] = dispatch_replay(sbx, cwd, exec_cap, replay_deadline_s, **replay_options)
        step("replay_dispatched_once", t, pid=rec["replay_pid"])
        while True:
            time.sleep(15)
            r = run(sbx, f"cat {REPLAY_DIR}/rc 2>/dev/null || true", timeout=60)
            if r.stdout.strip():
                break
            if time.time() - t > replay_deadline_s + 120:
                rec["source_retained_for_audit"] = True
                save()
                raise TimeoutError("replay completion not observed; finish later with --continue (never redispatch)")
        rc = r.stdout.strip()
        for name in ("replay_report.json", "replay_stdout.log"):
            (state_dir / f"{tag}-{name}").write_bytes(bytes(sbx.files.read(f"{REPLAY_DIR}/{name}", format="bytes")))
        rep = json.loads((state_dir / f"{tag}-replay_report.json").read_text())
        step("replay", t, rc=rc, summary={k: v for k, v in rep.items() if k not in ("results", "pi_output_manifest")})
        plan = json.loads((bundle / "plan.json").read_text())
        t = time.time()
        post_cut = probe_post_cut(sbx, run, plan, state_dir, tag)
        step("post_cut_probe", t, **post_cut)
        try:
            gate_result = validate_replay(plan, rep, rc, meta.get("replay_reviewed_exceptions", {}),
                                          policy=replay_policy, post_cut=post_cut)
        except RuntimeError as exc:
            if replay_policy == "lenient":
                rec["divergence_report"] = lenient_report(plan, rep, post_cut)
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
        if replay_policy == "lenient" and isinstance(gate_result, dict):
            # Per-task divergence report: travels with the bundle and the state so results can quote it.
            rec["divergence_report"] = gate_result
            (state_dir / f"{tag}-divergence_report.json").write_text(
                json.dumps(gate_result, ensure_ascii=False, indent=1) + "\n")
            try:
                (bundle / "divergence_report.json").write_text(
                    json.dumps(gate_result, ensure_ascii=False, indent=1) + "\n")
            except OSError:
                pass
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


def finish_prepared_source(state_path: Path, *, observe_timeout_s: int = 10800,
                           replay_policy: str | None = None) -> dict[str, Any]:
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

    if meta.get("agent") in ("pi", "codex"):
        check_native_replay(meta["agent"], Path(rec["task_dir"]), bundle, replay_policy or rec.get("replay_policy", "strict"))
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
    policy = replay_policy or rec.get("replay_policy") or "strict"
    t = time.time()
    post_cut = probe_post_cut(sb, run, plan, state_dir, rec["tag"])
    step("post_cut_probe", t, **post_cut)
    try:
        gate_result = validate_replay(plan, report, rc, meta.get("replay_reviewed_exceptions", {}),
                                      policy=policy, post_cut=post_cut)
    except RuntimeError as exc:
        if policy == "lenient":
            rec["divergence_report"] = lenient_report(plan, report, post_cut)
        rec["replay_gate_error"] = str(exc)
        rec["replay_gate_passed"] = False
        save()
        raise
    rec.pop("replay_gate_error", None)
    rec["replay_policy"] = policy
    if policy == "lenient" and isinstance(gate_result, dict):
        rec["divergence_report"] = gate_result
        (state_dir / f"{rec['tag']}-divergence_report.json").write_text(
            json.dumps(gate_result, ensure_ascii=False, indent=1) + "\n")
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
