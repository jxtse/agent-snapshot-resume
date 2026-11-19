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
  exec        Bash (foreground): `bash -lc` in the original cwd with timeout = max(60, 2*orig_duration+30), capped
              by --exec-cap. Output compared with the original observation (exact / normalized-whitespace /
              numeric-tolerant / differs).
  exec_background  Bash run_in_background or auto-backgrounded: executed in the foreground with
              timeout = max(60, 2*orig_runtime+30) when the original task finished before step k (its outputs feed
              the prefix; the output file gets the agent's completion trailer, plan "bg_output_trailer"); if it was
              still running at step k, it is started detached (nohup) and left running, matching the original state
              approximately. Flagged in report. Failed terminal jobs are bounded by their historical duration;
              known timeout/signal exits are reproduced, ordinary fast failures must actually recur.
              Jobs retained from a mid-run image have their cloned output writers stopped before a foreground rerun.
  pacing      commands that leave processes running (`nohup ... &`) use file capture, avoiding inherited pipe
              EOF waits. Track their isolated session, including process groups created by timeout. While any
              member lives, later Bash/Read actions wait until the original time gap since launch has elapsed
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


def run_bash(cmd, cwd, timeout, merge_stderr=False, termination_code=None, shell_env=None, pi_output_file=None, pi_bash=False, pi_user=None):
    """merge_stderr: one capture file for both streams, in write order (how Claude Code captures a background task's
    output file); stderr is then empty."""
    t0 = time.time()
    if shell_env and shell_env.get("PATH"):
        import shlex
        cmd = "export PATH=" + shlex.quote(shell_env["PATH"]) + "; " + cmd
    # Regular files decouple shell completion from EOF on inherited child descriptors.
    # Do not rewrite shell syntax: redirects, heredocs, pipelines and $! retain their semantics.
    import tempfile
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        shell = ["/bin/bash", "-c"] if pi_bash else ["bash", "-lc"]
        env = os.environ.copy()
        if pi_bash:
            for key in ("PI_SESSION_FILE", "PI_SESSION_ID", "PI_MODEL", "PI_PROVIDER", "PI_REASONING_LEVEL"):
                env.pop(key, None)
        env.update(shell_env or {})
        p = subprocess.Popen([*shell, cmd], env=env,
                             cwd=cwd if cwd and os.path.isdir(cwd) else None,
                             stdin=subprocess.DEVNULL, stdout=out_file,
                             stderr=out_file if merge_stderr else err_file, start_new_session=True)
        timed_out = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            sig = signal.SIGKILL if termination_code == 137 else signal.SIGTERM
            signal_session(p.pid, sig)
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                signal_session(p.pid, signal.SIGKILL)
                p.wait()
            # A child in a separate group may survive after the shell exits on TERM.
            signal_session(p.pid, signal.SIGKILL)
        def read_capture(f):
            # pread does not change the shared open-file offset used by surviving writers.
            return os.pread(f.fileno(), os.fstat(f.fileno()).st_size, 0).decode(errors="replace")
        proof = {}
        if pi_output_file:
            proof = restore_pi_log(pi_output_file, out_file, pi_user)
        rc = termination_code if timed_out else (128 - p.returncode if p.returncode < 0 else p.returncode)
        return {**proof, "rc": rc, "stdout": read_capture(out_file), "stderr": "" if merge_stderr else read_capture(err_file),
                "dur": round(time.time() - t0, 2), "timeout": timed_out and termination_code is None,
                "termination_reproduced": timed_out and termination_code is not None, "pgid": p.pid}


def session_groups(sid):
    """Live groups within our isolated Linux session, including timeout-created groups.

    GNU timeout creates a new process group without changing session. Zombies must not
    keep pacing alive. Outside Linux, retain the original process-group fallback.
    """
    if not os.path.isdir("/proc/self"):
        return None
    groups = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/" + name + "/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid and fields[0] not in ("Z", "X"):
                groups.add(int(fields[2]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return groups


def signal_session(sid, sig):
    groups = session_groups(sid)
    for pgid in ({sid} if groups is None else groups | {sid}):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass


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
    """True while the isolated session has live members (group fallback outside Linux)."""
    groups = session_groups(pgid)
    if groups is not None:
        return bool(groups)
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


PI_LOG_DIR = "/tmp"
PI_TRUNC_RE = re.compile(
    r"\n\n\[Showing (?:lines \d+-\d+ of \d+(?: \(50\.0KB limit\))?|last [\d.]+KB of line \d+ "
    r"\(line is [\d.]+[KM]B\))\. Full output: (/tmp/pi-bash-[0-9a-f]+\.log)\]"
    r"(?=(?:\n\nCommand (?:timed out after [\d.]+ seconds|exited with code \d+|aborted))?\s*\Z)")


def pi_log_path(orig):
    match = PI_TRUNC_RE.search(orig or "")
    return match[1] if match else None


def restore_pi_log(recorded, capture, user=None):
    import pwd
    import tempfile
    if not re.fullmatch(r"/tmp/pi-bash-[0-9a-f]+\.log", recorded):
        raise ValueError("invalid Pi output path")
    target = os.path.join(PI_LOG_DIR, os.path.basename(recorded))
    owner = pwd.getpwuid(os.geteuid()) if user is None else (
        pwd.getpwuid(int(user)) if str(user).isdecimal() else pwd.getpwnam(user))
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=PI_LOG_DIR, delete=False) as part:
        try:
            offset = 0
            size = os.fstat(capture.fileno()).st_size
            while offset < size:
                chunk = os.pread(capture.fileno(), min(1024 * 1024, size - offset), offset)
                if not chunk:
                    raise IOError("short Pi output capture")
                digest.update(chunk)
                part.write(chunk)
                offset += len(chunk)
            part.flush()
            if os.geteuid() != owner.pw_uid or os.fstat(part.fileno()).st_gid != owner.pw_gid:
                os.fchown(part.fileno(), owner.pw_uid, owner.pw_gid)
            os.fchmod(part.fileno(), 0o600)
            os.replace(part.name, target)
        finally:
            if os.path.exists(part.name):
                os.unlink(part.name)
    def assume_owner():
        os.initgroups(owner.pw_name, owner.pw_gid)
        os.setgid(owner.pw_gid)
        os.setuid(owner.pw_uid)
    # The digest is read by the account which will actually open this private file.
    check = subprocess.run(["shasum", "-a", "256", target] if sys.platform == "darwin" else
                           ["sha256sum", target], check=True, stdout=subprocess.PIPE,
                           preexec_fn=assume_owner if os.geteuid() != owner.pw_uid else None)
    if check.stdout.split()[0].decode() != digest.hexdigest():
        raise IOError("Pi output readback mismatch")
    return {"pi_output_file": recorded, "pi_output_sha256": digest.hexdigest(), "pi_output_reader_uid": owner.pw_uid}


def pi_truncated_view(orig, new):
    """Render Pi's observed tail from the full replay capture."""
    path = pi_log_path(orig)
    if not path:
        return new
    lines = (new[:-1] if new.endswith("\n") else new).split("\n")
    limit = 50 * 1024
    if len(lines) <= 2000 and len(new.encode()) <= limit:
        return new
    kept, size = [], 0
    by_bytes = False
    for line in reversed(lines):
        if len(kept) == 2000:
            break
        data = line.encode()
        cost = len(data) + bool(kept)
        if size + cost > limit:
            if not kept:
                tail = data[-limit:].decode(errors="ignore")
                total_bytes = len(data)
                unit = "KB" if total_bytes < 1024 ** 2 else "MB"
                divisor = 1024 if unit == "KB" else 1024 ** 2
                return (tail + f"\n\n[Showing last {len(tail.encode()) / 1024:.1f}KB of line {len(lines)} "
                        f"(line is {total_bytes / divisor:.1f}{unit}). Full output: {path}]")
            by_bytes = True
            break
        kept.append(line)
        size += cost
    total = len(lines)
    limit_note = " (50.0KB limit)" if by_bytes else ""
    return ("\n".join(reversed(kept)) + f"\n\n[Showing lines {total - len(kept) + 1}-{total} of {total}"
            f"{limit_note}. Full output: {path}]")


def stop_snapshot_background(action):
    """Stop cloned writers identified by the exact CC output fd before rerunning the action.

    The source cleanup stops CC itself, so its completion trailer cannot be relied upon. Linux /proc is
    required here: never silently duplicate a job when process ownership cannot be checked.
    """
    import glob
    if not os.path.isdir("/proc/self/fd") or not action.get("background_output_file"):
        raise RuntimeError("cannot identify cloned background writers")
    path = action["background_output_file"]
    groups = set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == path:
                pgid = os.getpgid(int(fd.split("/")[2]))
                if pgid == os.getpgrp():
                    raise RuntimeError("background writer shares replayer process group")
                groups.add(pgid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # All fd holders must stop before the output is replaced. Zombies hold no fds.
    deadline = time.monotonic() + 5
    while groups:
        holders = []
        for fd in glob.glob("/proc/[0-9]*/fd/*"):
            try:
                if os.readlink(fd) == path:
                    holders.append(fd)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not holders:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("cloned background writers did not stop")
        time.sleep(0.05)
    return sorted(groups)


def edit_original_spans(path, edits, originally_failed):
    """Pi matches each old text against the same input before writing any replacement."""
    try:
        with open(path, encoding="utf-8", newline="") as stream:
            source = stream.read()
    except FileNotFoundError:
        return "edit_failed", "error_reproduced" if originally_failed else "edit_target_missing"
    bom = "\ufeff" if source.startswith("\ufeff") else ""
    source = source[len(bom):]
    first_lf = source.find("\n")
    crlf = first_lf > 0 and source[first_lf-1] == "\r"
    def lf(text):
        return text.replace("\r\n", "\n").replace("\r", "\n")
    source = lf(source)
    import unicodedata
    def fuzzy(text):
        text = unicodedata.normalize("NFKC", text)
        text = "\n".join(line.rstrip("\u0009\u000b\u000c\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff") for line in text.split("\n"))
        for chars, plain in (("‘’‚‛", "'"), ("“”„‟", '"'), ("‐‑‒–—―−", "-")):
            text = text.translate(dict.fromkeys(map(ord, chars), plain))
        return text
    original = source
    fuzzy_original = fuzzy(source)
    use_fuzzy = any(lf(edit["old_string"]) not in source for edit in edits)
    if use_fuzzy:
        source = fuzzy_original
    spans = []
    for edit in edits:
        old = lf(edit["old_string"])
        if use_fuzzy and old not in source:
            old = fuzzy(old)
        start = source.find(old)
        # Pi counts occurrences in normalized space even when the first match was exact.
        needle = fuzzy(old)
        if not needle or start == -1 or fuzzy_original.count(needle) != 1:
            return "edit_failed", "error_reproduced" if originally_failed else "edit_target_missing"
        spans.append((start, start+len(old), lf(edit["new_string"])))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return "edit_failed", "error_reproduced" if originally_failed else "edit_ambiguous"
    def replace_ranges(text, replacements, offset=0):
        chunks = []
        pos = 0
        for start, end, replacement in replacements:
            chunks.extend((text[pos:start-offset], replacement))
            pos = end - offset
        return "".join([*chunks, text[pos:]])

    if use_fuzzy:
        # Pi normalizes an entire touched line, preserving original bytes on all other lines.
        from bisect import bisect_left, bisect_right
        from itertools import accumulate
        original_lines = re.findall(r"[^\n]*\n|[^\n]+$", original)
        normalized_lines = re.findall(r"[^\n]*\n|[^\n]+$", source)
        offsets = list(accumulate(map(len, normalized_lines), initial=0))
        groups = []
        for span in spans:
            first = bisect_right(offsets, span[0]) - 1
            stop = bisect_left(offsets, span[1])
            if groups and first < groups[-1][1]:
                groups[-1][1] = max(stop, groups[-1][1])
                groups[-1][2].append(span)
            else:
                groups.append([first, stop, [span]])
        chunks = []
        pos = 0
        for first, stop, edits_in_group in groups:
            chunks.extend(original_lines[pos:first])
            chunks.append(replace_ranges(source[offsets[first]:offsets[stop]], edits_in_group, offsets[first]))
            pos = stop
        source = "".join(chunks + original_lines[pos:])
    else:
        source = replace_ranges(source, spans)
    if source == original:
        return "edit_failed", "error_reproduced" if originally_failed else "edit_no_change"
    if originally_failed:
        return "skipped_orig_error", "orig_error_now_applicable"
    payload = (bom + (source.replace("\n", "\r\n") if crlf else source)).encode()
    with open(path, "wb") as stream:
        stream.write(payload)
    with open(path, "rb") as stream:
        if stream.read() != payload:
            raise IOError("Pi edit readback mismatch")
    return "applied", "exact"


def main():
    plan_p, rep_p = sys.argv[1], sys.argv[2]
    cap = 1800
    if "--exec-cap" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--exec-cap") + 1])
    pi_user = sys.argv[sys.argv.index("--pi-user") + 1] if "--pi-user" in sys.argv else None
    dry = "--dry" in sys.argv
    with open(plan_p) as f:
        plan = json.load(f)
    if plan.get("replay_refusals"):
        raise RuntimeError("; ".join(plan["replay_refusals"]))
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
        if live and not dry and tool in ("Bash", "Read"):
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
            elif tool in ("Edit", "MultiEdit") and args.get("match_original"):
                rec["status"], rec["fidelity"] = edit_original_spans(args["file_path"], args.get("edits", [args]), a.get("is_error"))
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
            elif tool == "Read" and a.get("evidence_read_for"):
                with open(args["file_path"], encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                original = a.get("orig_text") or ""
                rows = list(re.finditer(r"(?m)^(\s*(\d+)[→\t])(.*)$", original))
                if not rows:
                    raise RuntimeError("unsupported background Read rendering")
                def render(match):
                    index = int(match[2]) - 1
                    return match[1] + (lines[index] if index < len(lines) else "<missing line>")
                actual = re.sub(r"(?m)^(\s*(\d+)[→\t])(.*)$", render, original)
                rec.update(status="read", fidelity=compare(original, actual), stdout_full=actual,
                           stderr_full="", orig_full=original)
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
                    orig_to = original_timeout(a.get("orig_text")) if se == "exec" else None
                    termination_code = None
                    failed_bg = se == "exec_background" and a.get("bg_status") == "failed"
                    if failed_bg and od is not None and a.get("bg_exit_code") in (124, 137, 143, 144):
                        termination_code = a["bg_exit_code"]
                    if orig_to:  # original hit CC's Bash timeout: reproduce the same cut-off (state = partial run)
                        to = orig_to
                        rec["orig_timed_out"] = True
                    else:
                        to = int(min(cap, max(60, 2 * (od or 0) + 30)))
                    if failed_bg and od is not None:
                        to = min(cap, max(0.1, od + 0.25))
                    if termination_code and to < od:
                        termination_code = None  # the execution cap cannot masquerade as the historical kill
                    pi_bash = args.get("pi_bash")
                    pi_log = args.get("pi_output_file")
                    if pi_bash and args.get("timeout"):
                        original_limit = args["timeout"] / 1000
                        to = min(cap, original_limit)
                        if to == original_limit:
                            termination_code = 137
                    if scripts_measure_time(cmd, cwd):
                        rec["scripts_measure_time"] = True
                    if a.get("bg_inflight_at_snapshot"):
                        rec["snapshot_background_groups_stopped"] = stop_snapshot_background(a)
                        rec["background_resume"] = "rerun_from_snapshot"
                    capture = {"pi_bash": True, "pi_user": pi_user, "pi_output_file": pi_log} if pi_bash else {}
                    r = run_bash(cmd, cwd, to, merge_stderr=(se == "exec_background" or pi_bash),
                                 termination_code=termination_code, shell_env={
                                     **(plan.get("shell_env") or {}), **(args.get("pi_env") or {})}, **capture)
                    rec["shell_environment"] = ("captured_path" if plan.get("shell_env") else
                                                "nonlogin_shell" if pi_bash else "login_shell")
                    if r.get("termination_reproduced"):
                        rec["termination_reproduced"] = True
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
                        comparison_rc = r["rc"]
                        if pi_bash:
                            new = r["stdout"]
                            if new.startswith("\ufeff"):
                                new = new[1:]  # Pi's streaming TextDecoder strips the first BOM.
                            if pi_log:
                                rec.update(pi_output_file=r["pi_output_file"], pi_output_sha256=r["pi_output_sha256"],
                                           pi_output_reader_uid=r["pi_output_reader_uid"])
                                rec["pi_output_dimensions"] = {"totalBytes": len(new.encode()),
                                                               "totalLines": new.count("\n") + int(bool(new) and not new.endswith("\n"))}
                                historical = args.get("pi_historical_output") or {}
                                actual = dict(rec["pi_output_dimensions"], sha256=rec["pi_output_sha256"])
                                rec["pi_output_evidence"] = (
                                    "historical_mismatch" if any(actual[k] != v for k, v in historical.items())
                                    else "historical_sha256" if "sha256" in historical else
                                    "historical_dimensions" if historical else "replay_hash_only")
                                new = pi_truncated_view(a.get("orig_text"), new)
                            elif r["rc"] and not args.get("pi_timeout_status"):
                                # The IR canonicalizes Pi's exit trailer to the Bash error prefix.
                                new = new.rstrip("\n") or "(no output)"
                            if args.get("pi_timeout_status") and r.get("termination_reproduced"):
                                new = (new + "\n\n" if new else "") + (
                                    f"Command timed out after {args['pi_timeout_status']} seconds")
                                comparison_rc = None
                        rec["fidelity"] = compare(a.get("orig_text"), new, comparison_rc)
                        if pi_log:
                            rec["fidelity_scope"] = "displayed_observation"
                            if rec["pi_output_evidence"] == "historical_mismatch":
                                rec["fidelity"] = "pi_historical_output_mismatch"
                        if rec["fidelity"] in ("differs", "numbers_differ"):
                            rec["orig_head"] = (a.get("orig_text") or "")[:400]
                            rec["new_head"] = new[:400]
                    else:
                        rec["fidelity"] = "bg_rerun_foreground"
                        if failed_bg and r["rc"] == a.get("bg_exit_code"):
                            rec["fidelity"] = "bg_termination_reproduced" if termination_code else "bg_failure_reproduced"
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
