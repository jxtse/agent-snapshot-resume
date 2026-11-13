#!/usr/bin/env python3
"""In-container replayer: rebuild workspace state before resume step k from a replay plan.

Runs INSIDE the task sandbox (any python3 >= 3.8, stdlib only; it is uploaded as a single file, so it must not import
anything from this package). Input: plan JSON produced by `asr build-bundle` (actions in original timestamp order).
Output: fidelity report JSON.

Plan actions use a canonical tool vocabulary (Write / Edit / Bash / Agent / Task* / web) and Claude Code's observation
text conventions (e.g. "Exit code N" prefix on failed Bash). Adapters for other agents map their tool calls onto it.

Policy (per side-effect class):
  file_write  Write: write args.content exactly (mkdir -p parents).  Edit: apply old_string->new_string exactly
              (replace_all honoured); NotebookEdit: skipped+flagged. Originally-failed edits (is_error) are replayed
              too and must fail again (fidelity: 'error_reproduced').
  exec        Bash (foreground): `bash -c` in the original cwd with timeout = max(60, 2*orig_duration+30), capped
              by --exec-cap. Output compared with the original observation (exact / normalized-whitespace /
              numeric-tolerant / differs).
  exec_background  Bash run_in_background or auto-backgrounded: executed in the foreground with
              timeout = max(60, 2*orig_runtime+30) when the original task finished before step k (its outputs feed
              the prefix; the output file gets the agent's completion trailer, plan "bg_output_trailer"); if it was
              still running at step k, it is started detached (nohup) and left running, matching the original state
              approximately. Flagged in report.
  pacing      a replayed foreground command that leaves processes running (`nohup ... &`) keeps its process group
              alive; while it is, later Bash actions wait until the original time gap since that command has elapsed
              (capped by --exec-cap), so polls and log reads see a comparable progress point ("paced_wait_s"). If the
              wait cannot be kept (beyond the cap, or the action's original time is unknown), the action is marked
              "pacing_unmet" and the replay gate rejects the snapshot. A surviving job whose own original start time
              is unknown makes every later Bash action pacing_unmet.
  subagent    Agent/Task spawn: not replayed as a model call; the sidechain's own side-effecting tool calls that
              finished before step k are included in the plan (merged by timestamp) and replayed like main calls.
  task_mgmt   TaskCreate/TaskUpdate: restored from the session by the agent adapter (Claude Code:
              $CLAUDE_CONFIG_DIR/tasks/<sid>/<id>.json), not here. TaskStop: kills nothing (background replays are foreground); Cron*: ignored.
  web         not replayed (network may differ); flagged.
  read_only   skipped by default (no state change).
Usage: python3 replay_inside.py plan.json report.json [--exec-cap 1800] [--dry]
"""
import hashlib, json, os, re, signal, subprocess, sys, time

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
CWD_RESET = re.compile(r"\n?Shell cwd was reset to \S+\s*$")


def norm(s):
    s = ANSI.sub("", s or "")
    s = CWD_RESET.sub("", s)
    return "\n".join(l.rstrip() for l in s.strip().splitlines())


def numfree(s):
    return NUM.sub("#", norm(s))


VOLATILE = [re.compile(p) for p in (r"(?m)^(?:real|user|sys)\s+\d+m\d+(?:\.\d+)?s$", r"0x[0-9a-fA-F]{6,}", r"\b[A-Z][a-z]{2} +\d{1,2} +(\d{2}:\d{2}|\d{4})\b",
                                    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?", r"\d+(\.\d+)?\s?(s|ms|sec|seconds)\b",
                                    r"time=\S+", r"elapsed\S*\s*\S+", r"(?m)^total \d+$", r"\bpid[= ]\d+", r"(?m)^(?:real|user|sys)\s+\d+(?:\.\d+)?$")]


def volatile_free(s):
    s = norm(s)
    for p in VOLATILE:
        s = p.sub("<V>", s)
    return s


NO_OUTPUT = "(Bash completed with no output)"  # what Claude Code shows for an empty successful Bash result


def compare(orig, new, rc=None):
    if orig is None:
        return "no_original"
    if rc not in (None, 0) and not new.startswith("Exit code"):
        new = f"Exit code {rc}\n" + new  # CC prefixes failed Bash results
    if rc in (None, 0) and not new.strip() and norm(orig) == NO_OUTPUT:
        return "exact"
    if orig == new:
        return "exact"
    if norm(orig) == norm(new):
        return "normalized"
    if sorted(norm(orig).splitlines()) == sorted(norm(new).splitlines()):
        return "same_lines_reordered"  # stdout/stderr interleaving differs
    if volatile_free(orig) == volatile_free(new):
        return "volatile_only"  # addresses / dates / timings
    # Combine only address masking with line-order normalization; keep numeric values.
    address_norm = lambda s: sorted(re.sub(r"0x[0-9a-fA-F]{6,}", "<ADDR>", norm(s)).splitlines())
    if address_norm(orig) == address_norm(new):
        return "address_and_order_only"
    aligned = lambda s: sorted(line.strip() for line in norm(s).splitlines() if line.strip())
    if aligned(orig) == aligned(new):
        return "line_alignment_and_order_only"
    if numfree(orig) == numfree(new):
        return "numbers_differ"
    if orig.startswith("<persisted-output>"):
        return "persisted_preview"
    return "differs"


def original_timeout(text):
    match = re.search(r'^Exit code 143\nCommand timed out after (\d+)m (\d+)s(?:\n|$)', text or '')
    return int(match[1]) * 60 + int(match[2]) if match else None


def run_bash(cmd, cwd, timeout, merge_stderr=False):
    """merge_stderr: one pipe for both streams, in write order (how Claude Code captures a background task's
    output file); stderr is then empty."""
    t0 = time.time()
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd if cwd and os.path.isdir(cwd) else None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                         start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
        out = stdout.decode(errors="replace")
        err = (stderr or b"").decode(errors="replace")
        return {"rc": p.returncode, "stdout": out, "stderr": err, "dur": round(time.time() - t0, 2), "timeout": False,
                "pgid": p.pid}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = p.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
        out = (stdout or b"").decode(errors="replace")
        err = (stderr or b"").decode(errors="replace")
        return {"rc": None, "stdout": out, "stderr": err, "dur": round(time.time() - t0, 2), "timeout": True}


TIMING_SRC = re.compile(r"time\.time\(|perf_counter|time\.monotonic|process_time|timeit")


def scripts_measure_time(cmd, cwd):
    """Python scripts the command runs (python x.py) whose source measures wall time, read from this sandbox."""
    for path in re.findall(r"\bpython3?\s+(?:-\w+\s+)*([\w./-]+\.py)\b", cmd or ""):
        p = path if path.startswith("/") else os.path.join(cwd or "", path)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                if TIMING_SRC.search(f.read(500_000)):
                    return True
        except OSError:
            continue
    return False


def group_alive(pgid):
    """True while any process of the replayed command's process group (e.g. a `nohup ... &` child) is alive."""
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def iso_ts(s):
    import datetime as dt
    try:
        return dt.datetime.fromisoformat((s or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def pace_wait(live, a, cap):
    """Shell background jobs started by earlier replayed commands are still running: keep the original pacing, i.e.
    do not start action a before (its original start - the job's original start) has elapsed since the job's replay
    start. Returns (seconds waited, met). met is False if the wait would exceed cap or a's original time is unknown
    while a job is alive; the gate then rejects the replay (the read could see another progress point)."""
    t = iso_ts(a.get("timestamp"))
    waited = 0.0
    for job in list(live):
        if not group_alive(job["pgid"]):
            live.remove(job)
            continue
        if t is None or job.get("orig_t") is None:
            return round(waited, 1), False
        target = job["replay_t"] + max(0.0, t - job["orig_t"])
        if target - time.time() > cap - waited:
            return round(waited, 1), False
        while time.time() < target and group_alive(job["pgid"]):
            step = min(1.0, max(0.0, target - time.time()))
            time.sleep(step)
            waited += step
    return round(waited, 1), True


def cc_bash_text(r):
    """Approximate CC's tool_result text for Bash: stdout, then stderr on a new line."""
    out, err = r["stdout"].rstrip("\n"), r["stderr"].rstrip("\n")
    if out and err:
        return out + "\n" + err
    return out or err


def main():
    plan_p, rep_p = sys.argv[1], sys.argv[2]
    cap = 1800
    if "--exec-cap" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--exec-cap") + 1])
    dry = "--dry" in sys.argv
    with open(plan_p) as f:
        plan = json.load(f)
    results = []
    t_start = time.time()
    shell_cwd = plan.get("initial_cwd") or "/app"
    live = []  # shell background jobs (nohup ... &) of replayed commands still running: keep original pacing
    trailer = plan.get("bg_output_trailer")
    for n, a in enumerate(plan["actions"]):
        tool, se, args = a["tool"], a["side_effect"], a["args"]
        rec = {"n": n, "call_id": a["call_id"], "tool": tool, "side_effect": se, "origin": a.get("origin"),
               "atif_step_id": a.get("atif_step_id"), "orig_is_error": a.get("is_error"),
               "orig_duration_s": a.get("duration_s")}
        rec["started_at"] = round(time.time(), 3)  # replay clock, for the gate's clock-provenance check
        if live and not dry and tool == "Bash":
            w, met = pace_wait(live, a, cap)
            if w:
                rec["paced_wait_s"] = w
            if not met:
                rec["pacing_unmet"] = True
        try:
            if dry:
                rec["status"] = "dry"
            elif tool == "Write" and a.get("is_error"):
                rec["status"] = "skipped_orig_error"
                rec["fidelity"] = "orig_error_not_applied"
            elif tool == "Write":
                fp = args["file_path"]
                os.makedirs(os.path.dirname(fp) or "/", exist_ok=True)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(args.get("content", ""))
                rec["status"] = "applied"
                rec["fidelity"] = "exact" if not a.get("is_error") else "orig_error_but_applied"
            elif tool == "Edit":
                fp = args["file_path"]
                old, new = args.get("old_string", ""), args.get("new_string", "")
                try:
                    txt = open(fp, encoding="utf-8").read()
                except FileNotFoundError:
                    txt = None
                if txt is None or old not in txt:
                    rec["status"] = "edit_failed"
                    rec["fidelity"] = "error_reproduced" if a.get("is_error") else "edit_target_missing"
                else:
                    cnt = txt.count(old)
                    if cnt > 1 and not args.get("replace_all"):
                        rec["status"] = "edit_failed_ambiguous"
                        rec["fidelity"] = "error_reproduced" if a.get("is_error") else "edit_ambiguous"
                    elif a.get("is_error"):
                        rec["status"] = "skipped_orig_error"
                        rec["fidelity"] = "orig_error_now_applicable"
                    else:
                        txt = txt.replace(old, new) if args.get("replace_all") else txt.replace(old, new, 1)
                        open(fp, "w", encoding="utf-8").write(txt)
                        rec["status"] = "applied"
                        rec["fidelity"] = "exact"
            elif tool == "Bash" and se in ("exec", "exec_background"):
                cmd = args.get("command", "")
                cwd = a.get("cwd") or shell_cwd
                if se == "exec_background" and not a.get("bg_finished_before_k", True):
                    lg = f"/tmp/.asr-replay-bg-{a['call_id'][-8:]}.log"
                    subprocess.Popen(["bash", "-c", f"nohup bash -c {json.dumps(cmd)} > {lg} 2>&1 &"],
                                     cwd=cwd if os.path.isdir(cwd) else None)
                    rec["status"] = "started_detached"
                    rec["fidelity"] = "bg_running_at_k"
                else:
                    od = a.get("bg_duration_s") if se == "exec_background" else a.get("duration_s")
                    orig_to = original_timeout(a.get("orig_text"))
                    if orig_to:  # original hit CC's Bash timeout: reproduce the same cut-off (state = partial run)
                        to = orig_to
                        rec["orig_timed_out"] = True
                    else:
                        to = int(min(cap, max(60, 2 * (od or 0) + 30)))
                    if scripts_measure_time(cmd, cwd):
                        rec["scripts_measure_time"] = True
                    r = run_bash(cmd, cwd, to, merge_stderr=(se == "exec_background"))
                    rec.update({"status": "timeout" if r["timeout"] else "ran", "rc": r["rc"], "dur_s": r["dur"],
                                "timeout_s": to})
                    if not r["timeout"] and r.get("pgid") and group_alive(r["pgid"]):
                        # orig_t None (no original start time): any later action while it lives is pacing_unmet
                        live.append({"pgid": r["pgid"], "replay_t": time.time() - r["dur"],
                                     "orig_t": iso_ts(a.get("timestamp")), "call_id": a["call_id"]})
                        rec["left_background_job"] = True
                    rec["stdout_full"] = r["stdout"]
                    rec["stderr_full"] = r["stderr"]
                    rec["orig_full"] = a.get("orig_text")
                    if se == "exec":
                        new = cc_bash_text(r)
                        rec["fidelity"] = compare(a.get("orig_text"), new, r["rc"])
                        if rec["fidelity"] in ("differs", "numbers_differ"):
                            rec["orig_head"] = (a.get("orig_text") or "")[:400]
                            rec["new_head"] = new[:400]
                    else:
                        rec["fidelity"] = "bg_rerun_foreground"
                        output = a.get("background_output_file")
                        if output:
                            os.makedirs(os.path.dirname(output), exist_ok=True)
                            payload = r["stdout"] + r["stderr"]
                            if trailer and r["rc"] is not None:
                                payload += trailer.format(rc=r["rc"])  # e.g. CC 2.1.280: "\n[exited with code 0]\n"
                            payload = payload.encode()
                            with open(output, "wb") as f:
                                f.write(payload)
                            rec["background_output_file"] = output
                            rec["background_output_sha256"] = hashlib.sha256(payload).hexdigest()
                            rec["background_output_provenance"] = "actual replay output; historical full file unavailable"
                    if r["timeout"]:
                        rec["fidelity"] = "timeout_reproduced" if rec.get("orig_timed_out") else "timeout"
            elif tool in ("Agent", "Task", "SendMessage"):
                rec["status"] = "skipped"
                rec["fidelity"] = "subagent_model_call_not_replayed"
            elif se == "task_mgmt":
                rec["status"] = "skipped"
                rec["fidelity"] = "task_mgmt_restored_from_session" if tool in ("TaskCreate", "TaskUpdate") else "task_mgmt_ignored"
            elif se == "web":
                rec["status"] = "skipped"
                rec["fidelity"] = "web_not_replayed"
            else:
                rec["status"] = "skipped"
                rec["fidelity"] = "unsupported_tool"
        except Exception as e:  # noqa: BLE001
            rec["status"] = "replayer_exception"
            rec["error"] = repr(e)[:300]
            rec["fidelity"] = "replayer_exception"
        results.append(rec)
        sys.stdout.write(json.dumps({k: rec.get(k) for k in ("n", "tool", "status", "fidelity", "dur_s")}) + "\n")
        sys.stdout.flush()
    summ = {}
    for r in results:
        k = f"{r['tool']}:{r.get('fidelity')}"
        summ[k] = summ.get(k, 0) + 1
    exec_r = [r for r in results if r["side_effect"] == "exec"]
    rep = {"n_actions": len(results), "replay_wall_s": round(time.time() - t_start, 1), "summary": summ,
           "exec_total": len(exec_r),
           "exec_exact_or_normalized": sum(r.get("fidelity") in ("exact", "normalized") for r in exec_r),
           "exec_equivalent_modulo_volatile": sum(r.get("fidelity") in ("exact", "normalized", "same_lines_reordered", "volatile_only") for r in exec_r),
           "exec_numbers_differ": sum(r.get("fidelity") == "numbers_differ" for r in exec_r),
           "exec_differs": sum(r.get("fidelity") == "differs" for r in exec_r),
           "exec_timeouts": sum(r.get("fidelity") == "timeout" for r in exec_r),
           "exec_timeout_reproduced": sum(r.get("fidelity") == "timeout_reproduced" for r in exec_r),
           "file_write_total": sum(r["side_effect"] == "file_write" for r in results),
           "file_write_ok": sum(r["side_effect"] == "file_write" and r.get("fidelity") in ("exact", "error_reproduced", "orig_error_not_applied") for r in results),
           "results": results}
    with open(rep_p, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "results"}))


if __name__ == "__main__":
    main()
