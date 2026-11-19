"""Fail-closed gates for source replay and restored task/auxiliary files.

validate_replay(plan, report, rc, reviewed_exceptions) raises unless every planned action replayed with an accepted
fidelity. Two generic classes are accepted without review (`automatic_equivalence`: ls `.`/`..` rows, timing lines of
commands that measure wall time). Other outputs that differ only in known-benign ways (hardware/memory probes, shell clocks, cProfile timings,
terminal `| grep` no-match exit codes, original timeouts) are accepted ONLY when the bundle lists a reviewed exception
bound to the exact command's sha256 (bundle.json `replay_reviewed_exceptions`: {call_id: {kind, command_sha256, ...}}).
"""

def profile_without_times(text):
    """Parse a cProfile table, retaining science, call counts and function identities."""
    import re
    from .replay_inside import norm
    text = norm(text)
    header = re.search(r'(?m)^\s*(\d+) function calls \((\d+) primitive calls\) in \d+(?:\.\d+)? seconds$', text)
    if header is None:
        return None
    science = text[:header.start()].strip()
    tail = text[header.end():].strip().splitlines()
    tail = [line.strip() for line in tail if line.strip()]
    if len(tail) < 4 or tail[0] != 'Ordered by: cumulative time':
        return None
    count = re.fullmatch(r'List reduced from (\d+) to (\d+) due to restriction <(\d+)>', tail[1])
    if count is None or count[2] != count[3] or len(tail[3:]) != int(count[2]):
        return None
    if tail[2].split() != ['ncalls','tottime','percall','cumtime','percall','filename:lineno(function)']:
        return None
    rows = []
    for row in tail[3:]:
        parsed = re.fullmatch(r'(\d+(?:/\d+)?)\s+(?:\d+\.\d+\s+){4}(.+)', row)
        if parsed is None:
            return None
        rows.append(parsed.groups())
    return science, header.groups(), count.groups(), sorted(rows)


import re  # noqa: E402

WRITES = re.compile(r"open\([^)]*['\"][wax]b?\+?['\"]|\.to_(?:csv|parquet|pickle|json|hdf|feather)\(|savefig|"
                    r"np\.save|\.write(?:_text|_bytes)?\(|\.dump\(|mkdir|>\s*\S|\btee\b|\brm\b|\bmv\b|\bcp\b|"
                    r"shutil|os\.remove|unlink|\bsed\s+-i|pip3?\s+install")


# `ls -l` row: (permissions links owner group size) (date) (name, kept exactly)
LS_ROW = re.compile(r"^([-dlcbps][-rwxsStT]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+[\d,]+)\s+"
                    r"([A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4}))\s(.+)$")


TIMING_CALL = r"time\.time\(|perf_counter|time\.monotonic|process_time|timeit|datetime\.now\("


def _replayed_text(result):
    """The tool_result text Claude Code would have shown for the replayed command (see replay_inside)."""
    from .replay_inside import cc_bash_text
    new = cc_bash_text({"stdout": result.get("stdout_full", ""), "stderr": result.get("stderr_full", "")})
    rc = result.get("rc")
    if rc not in (None, 0) and not new.startswith("Exit code"):
        new = f"Exit code {rc}\n" + new
    return new


def _heredoc_split(command):
    """(shell text, heredoc bodies) of a command; the bodies of `<<'EOF' ... EOF` blocks are program text."""
    import re
    shell, bodies, rest = [], [], command
    while True:
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n", rest)
        if not m:
            shell.append(rest)
            break
        shell.append(rest[:m.end()])
        end = re.search(r"(?m)^\s*" + re.escape(m.group(1)) + r"\s*$", rest[m.end():])
        if not end:
            bodies.append(rest[m.end():])
            break
        bodies.append(rest[m.end():m.end() + end.start()])
        rest = rest[m.end() + end.end():]
    return "".join(shell), "\n".join(bodies)


def _stderr_placement_equal(action, result, f):
    """The original text equals the replay's stdout IN ORDER once the replay's stderr lines are taken out of it
    (Claude Code interleaves stderr differently from `stdout + stderr`), with every line mapped through f."""
    from .replay_inside import norm
    f = f or (lambda l: l)
    old = [f(l) for l in norm(action.get("orig_text") or "").splitlines()]
    rc = result.get("rc")
    out = norm(result.get("stdout_full", "")).splitlines()
    err = [f(l) for l in norm(result.get("stderr_full", "")).splitlines() if l.strip()]
    if rc not in (None, 0) and old and old[0] == f(f"Exit code {rc}"):
        old = old[1:]
    pending = list(err)
    rest = []
    for l in old:
        if l in pending:
            pending.remove(l)
        else:
            rest.append(l)
    strip = lambda ls: [x for x in ls if x.strip()]
    return not pending and strip(rest) == strip([f(l) for l in out])


def automatic_equivalence(action, result):
    """Generic, review-free acceptance classes for replay outputs that differ only in ways that cannot reflect a
    different task state. Returns a reason string or None. Deliberately minimal; everything else (timings, profiler
    order, grep exit codes, mtimes, listing order, failed background tasks) needs a command-bound reviewed
    exception (validate_replay) or fails closed:
      ls_dot_entries_only   `ls -l*` listings whose only difference is the `.`/`..` rows and `total` block counts
                            (some sandbox filesystems omit them); every other row identical up to metadata-column
                            padding, file names exact.
      shell_diagnostic_prefix_only
                            bash's own error prefix differs (`/bin/bash: eval: line 18:` vs `bash: -c: line 9:`);
                            the original equals the replay's stdout in order once the replay's stderr lines are
                            taken out (stderr placement is the only other freedom).
    """
    import re
    from .replay_inside import norm
    if result.get("status") != "ran" or result.get("timeout"):
        return None
    command = (action.get("args") or {}).get("command", "")
    if bool(result.get("rc")) != bool(action.get("is_error")):
        return None
    old = norm(action.get("orig_text") or "").splitlines()
    new = norm(_replayed_text(result)).splitlines()
    ls_dot = re.compile(r"^d\S+\s+\d+\s+\S+\s+\S+\s+\d+\s+\S+\s+\d+\s+\S+\s\.{1,2}$")
    if re.search(r"\bls\s+-\w*l", command):
        def row(l):  # metadata columns compared up to padding; the file name (rest of the line) exactly
            m = LS_ROW.match(l)
            return (tuple(m.group(1).split()), m.group(2), m.group(3)) if m else ("line", l)
        strip = lambda ls: [row(l) for l in ls if not ls_dot.fullmatch(l) and not re.fullmatch(r"total \d+", l)]
        if strip(old) == strip(new) and strip(old) and any(ls_dot.fullmatch(l) for l in old + new):
            return "ls_dot_entries_only"
    # bash diagnostics carry the shell's own location prefix, which depends on how the command was invoked
    # (Claude Code: `/bin/bash: eval: line 18:`, replay: `bash: -c: line 9:`)
    diag = re.compile(r"^(?:/bin/|/usr/bin/)?bash:(?: eval:| -c:)? line \d+: ")
    if any(diag.match(l) for l in old + new) and _stderr_placement_equal(
            action, result, lambda l: diag.sub("bash: line <n>: ", l)):
        return "shell_diagnostic_prefix_only"
    return None


MASK_VERSION = 1
# numbers only as standalone tokens: `netCDF4` / `python3.11` stay readable, `0.1951` / `1.2e-05` are masked
_NUM = re.compile(r"(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![\w.])")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_DATE = re.compile(r"(?i)(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4})\b|"
                   r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?\b|"
                   r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} \w{3} \d{4}\b)")
_ADDR = re.compile(r"0x[0-9a-fA-F]{6,}")
_CLOCK = re.compile(r"^(?:real|user|sys)\s+(?:\d+m\d+(?:\.\d+)?s|\d+(?:\.\d+)?)$")
_DOT_ROW = re.compile(r"^d\S+\s+\d+\s+\S+\s+\S+\s+\d+\s+\S+\s+\d+\s+\S+\s+\.{1,2}$")
_MEM_ROW = re.compile(r"^Mem:\s+(?:\d+\s*)+\S*$")
_CPU_ROW = re.compile(r"^(?:CPU\(s\):|Model name:\s+).*$")
_FS_ROW = re.compile(r"^\S+\s+\d+(?:\.\d+)?[A-Za-z]?\s+\d+(?:\.\d+)?[A-Za-z]?\s+\d+(?:\.\d+)?[A-Za-z]?\s+\d+%\s+/\S*$")


def mask_volatile(text):
    """Normalise one replay output down to its information content (MASK_VERSION 1).

    Removed/neutralised: shell timing lines, `total N` block counts, `./..` listing rows, dates and timestamps,
    hex object addresses, machine-spec rows (`Mem:`, `df` mounts, `CPU(s)`, host CPU model), every number.
    Nothing about file names, paths, sizes-as-names, module names, exception types or any other text is touched, so
    two outputs that mask equal carry the same readable information; the numbers still have to be justified by the
    reviewed exception (changed_numbers_reused_later)."""
    from .replay_inside import norm
    out = []
    for line in norm(text).splitlines():
        line = line.rstrip()
        if not line.strip() or _CLOCK.fullmatch(line.strip()) or _DOT_ROW.fullmatch(line.strip()):
            continue
        s = line.strip()
        if re.fullmatch(r"total \d+", s):
            continue
        if _MEM_ROW.fullmatch(s):
            out.append("<mem>")
            continue
        if _CPU_ROW.fullmatch(s):
            out.append("<cpu>")
            continue
        if _FS_ROW.fullmatch(s):
            out.append("<fs>")
            continue
        line = re.sub(r"(?:^|(?<= ))(?:/bin/)?bash: (?:-c: line|eval: line|line) \d+:", "bash: <line>:", line)
        line = _UUID.sub("<uuid>", line)
        line = re.sub(r"\b(?:tbs|asr)-proxy\b", "<proxy-dir>", line)  # this harness's own proxy shim, renamed per campaign
        line = re.sub(r"[ \t]+", " ", _NUM.sub("<n>", _DATE.sub("<date>", _ADDR.sub("<addr>", line)))).strip()
        out.append(line)
    return out


def masked_equal(orig_text, replayed_text, order="ordered"):
    """Masked outputs are equal: in order (`ordered`), as a multiset (`sorted`, filesystem listing order), or with
    one side a prefix-subsequence of the other (`subsequence`, a listing cut short by `head` after extra `./..`
    rows)."""
    a, b = mask_volatile(orig_text), mask_volatile(replayed_text)
    if not a and not b:
        return False
    if order == "sorted":
        return sorted(a) == sorted(b)
    if order == "subsequence":
        it = iter(b)  # a must appear in b in the same relative order
        return all(x in it for x in a) or (lambda it2: all(x in it2 for x in b))(iter(a))
    return a == b


def replay_output_sha256(result):
    """Identity of one replayed command's observable outcome (for reviewed_replay_output exceptions)."""
    import hashlib
    data = "\0".join([result.get("stdout_full") or "", result.get("stderr_full") or "", str(result.get("rc"))])
    return hashlib.sha256(data.encode()).hexdigest()


# --------------------------------------------------------------------------------------------- replay policy
# The strict policy (validate_replay below) rejects any observation it cannot explain. The lenient policy accepts
# output divergences (numbers, package versions, listing order, external content, timing) because the resumed agent's
# conversation history is restored verbatim from the original session: a differing printed number is invisible to it.
# What it does not accept is a difference the agent will act on after the cut, so it blocks exactly three things:
#   (1) a replayed command fails where the original succeeded (non-zero rc, timeout, replayer crash, kill/OOM) or an
#       edit the original applied is not applied;
#   (2) a file the agent reads AFTER the cut (paths taken from the original session's post-cut tool calls) is missing
#       in the replayed environment;
#   (3) a background job still in flight at the cut, whose output the agent uses afterwards.
# Every accepted divergence is recorded (lenient_report) so a batch can report it next to the results.
LENIENT_ACCEPTED = {"exact", "normalized", "error_reproduced", "orig_error_not_applied", "bg_rerun_foreground",
                    "same_lines_reordered", "volatile_only", "address_and_order_only",
                    "line_alignment_and_order_only", "numbers_differ", "differs", "persisted_preview"}
KILL_MARKS = re.compile(r"(?i)\b(killed|out of memory|oom[-_ ]kill|oom_reaper|memoryerror|cannot allocate memory|"
                        r"std::bad_alloc|segmentation fault|core dumped|bus error|illegal instruction)\b")
KILL_RCS = (137, -9, 134, 139, 135)
# package/interpreter versions are dotted x.y.z (two or more dots); a plain decimal such as 0.831 is a number
_VERSION = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_NUMTOK = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _violently_killed(action, result):
    """A replay result whose process was killed by the kernel/scheduler (OOM, signal) rather than exiting.

    A kill mark in the output only counts when the original output did not already contain it: a command that
    legitimately prints or reproduces such a line (a test of the error path, a reproduced traceback) must not be
    mistaken for a killed process."""
    if result.get("rc") in KILL_RCS:
        return True
    if result.get("rc") == 0 and result.get("fidelity") in ("exact", "normalized"):
        return False
    text = (result.get("stdout_full") or "") + "\n" + (result.get("stderr_full") or "")
    orig = action.get("orig_text") or ""
    marks = {m.group(0).lower() for m in KILL_MARKS.finditer(text)}
    return bool(marks - {m.group(0).lower() for m in KILL_MARKS.finditer(orig)})


def _version_pairs(action, result):
    """Version tokens that differ, paired by position, between the original and the replayed output."""
    old = _VERSION.findall(action.get("orig_text") or "")
    new = _VERSION.findall(result.get("stdout_full") or "")
    if len(old) != len(new):
        return []
    return [[o, n] for o, n in zip(old, new) if o != n]


def lenient_divergence(action, result):
    """How one replayed action's observation diverged from the original (None when it did not)."""
    fidelity = result.get("fidelity")
    if fidelity in ("exact", "normalized", "error_reproduced", "orig_error_not_applied"):
        return None
    versions = _version_pairs(action, result)
    if versions and fidelity in ("numbers_differ", "differs", "volatile_only"):
        kind = "version_diff"
    elif fidelity == "numbers_differ":
        kind = "changed_numbers"
    elif fidelity in ("same_lines_reordered", "line_alignment_and_order_only", "address_and_order_only"):
        kind = "listing_or_line_order"
    elif fidelity == "volatile_only":
        kind = "timing_or_address"
    elif fidelity == "differs":
        kind = "text_differs"
    elif fidelity == "bg_rerun_foreground":
        # the background job's output file is rewritten with the replay's own output: its body may differ, its exit
        # status is checked separately
        kind = "background_content"
    else:
        kind = fidelity or "unknown"
    entry = {"call_id": action.get("call_id"), "tool": action.get("tool"), "side_effect": action.get("side_effect"),
             "fidelity": fidelity, "kind": kind}
    if versions:
        entry["versions"] = versions
    old_nums, new_nums = _NUMTOK.findall(action.get("orig_text") or ""), _NUMTOK.findall(result.get("stdout_full") or "")
    if len(old_nums) == len(new_nums):
        entry["changed_numbers"] = sum(1 for a, b in zip(old_nums, new_nums) if a != b)
    return entry


HARNESS_PATHS = ("/opt/tbs-proxy", "/opt/asr-proxy", "/logs/agent", "/tmp/.asr-resume")


def harness_path_only(action, result, paths=HARNESS_PATHS):
    """Only a terminal literal ls of a harness directory may disappear; preceding output is unchanged."""
    import shlex
    command = (action.get("args") or {}).get("command", "")
    clean = re.sub(r"2>\s*/dev/null", "", command)
    if re.search(r"[<>$`&()\n]", clean) or WRITES.search(clean):
        return False
    parts = clean.split(";")
    try:
        last = shlex.split(parts[-1])
        if len(last) != 2 or last[0] != "ls" or last[1] not in paths:
            return False
        marker = shlex.split(parts[-2]) if len(parts) > 1 else []
        if len(marker) != 2 or marker[0] != "echo" or not marker[1]:
            return False
        for part in parts[:-1]:
            for stage in part.split("|"):
                tokens = shlex.split(stage)
                if not tokens or not (tokens[0] in ("echo", "head") or tokens[:2] == ["pip", "list"]):
                    return False
    except ValueError:
        return False
    if result.get("rc") != 2 or result.get("status") != "ran":
        return False
    err = result.get("stderr_full", "").strip()
    if err and err != "ls: cannot access '" + last[1] + "': No such file or directory":
        return False
    old = (action.get("orig_text") or "").strip()
    new = result.get("stdout_full", "").strip()
    if not new.endswith("\n" + marker[1]) or not old.startswith(new + "\n"):
        return False
    tail = old[len(new):].strip()
    return bool(tail and all(re.fullmatch(r"[\w. -]+", line) for line in tail.splitlines()))


def _numeric_background_read(action, result, producer):
    """Lenient-only numeric drift in a direct read of this failed task's log."""
    from .replay_inside import NO_OUTPUT, numfree
    path = producer.get("background_output_file")
    if not path or result.get("status") != "ran" or result.get("rc") not in (None, 0):
        return False
    if result.get("pacing_unmet") or result.get("stderr_full"):
        return False
    args = action.get("args") or {}
    if action.get("tool") == "Read":
        if args.get("file_path") != path:
            return False
    elif action.get("tool") == "Bash":
        # Only literal log reads, optionally delayed or piped through head/tail.
        reader = r"(?:cat|head(?: -[0-9]+)?|tail(?: -[0-9]+)?)"
        command = args.get("command", "").strip()
        pattern = (r"(?:sleep [0-9]+;\s*)?" + reader + r" " + re.escape(path)
                   + r"(?: 2>/dev/null)?(?: \| (?:head|tail)(?: -[0-9]+)?)?")
        if not re.fullmatch(pattern, command):
            return False
    else:
        return False
    old = action.get("orig_text")
    if old is None or "stdout_full" not in result:
        return False
    new = result["stdout_full"]
    # Foreground reproduction exposes the recorded terminal trailer to earlier polls.
    trailer = r"\n?\[exited with code " + str(producer["bg_exit_code"]) + r"\]\s*$"
    old = re.sub(trailer, "", old)
    new = re.sub(trailer, "", new)
    if old == NO_OUTPUT:
        old = ""
    return numfree(old) == numfree(new)


def background_failure_reasons(action, result, results, *, actions=(), lenient=False):
    """Explain each failed equivalence condition without hiding later consumer failures."""
    reasons = []
    if action.get("bg_status") != "failed" or action.get("bg_exit_code") in (None, 0):
        reasons.append({"condition": "unknown_terminal_failure"})
    if result.get("status") != "ran" or result.get("rc") != action.get("bg_exit_code"):
        reasons.append({"condition": "terminal_outcome", "expected_rc": action.get("bg_exit_code"),
                        "actual_rc": result.get("rc"), "status": result.get("status")})
    if "bg_original_output" in action:
        from .replay_inside import numfree
        original = numfree(re.sub(r"\n?\[exited with code \d+\]\s*$", "", action["bg_original_output"]))
        actual = numfree(result.get("stdout_full", "") + result.get("stderr_full", ""))
        if bool(original) != bool(actual) or not (original.startswith(actual) or actual.startswith(original)):
            reasons.append({"condition": "output_prefix"})
    by_id = {r.get("call_id"): r for r in results}
    action_by_id = {a.get("call_id"): a for a in actions}
    for cid in action.get("bg_output_read_later") or []:
        read = by_id.get(cid)
        if read and read.get("fidelity") in ("exact", "normalized"):
            continue
        if (lenient and read and action.get("bg_exit_code") in (124, 137, 143, 144)
                and _numeric_background_read(action_by_id.get(cid, {}), read, action)):
            continue
        reasons.append({"condition": "consumer_not_exact", "call_id": cid,
                        "fidelity": (read or {}).get("fidelity")})
    return reasons


def background_failure_equivalent(action, result, results, *, actions=(), lenient=False):
    return not background_failure_reasons(action, result, results, actions=actions, lenient=lenient)


def recorded_no_match(action, result, *, lenient=False):
    """CC explicitly recorded grep's exit-1 convention; require unchanged complete output."""
    from .replay_inside import numfree
    same = numfree if lenient else lambda text: str(text).strip()
    return (action.get("tool") == "Bash" and action.get("side_effect") == "exec"
            and action.get("return_code_interpretation") == "No matches found"
            and not action.get("is_error") and result.get("status") == "ran"
            and result.get("rc") == 1 and not result.get("pacing_unmet")
            and action.get("orig_text") is not None
            and same(action["orig_text"]) == same(_replayed_text(dict(result, rc=0))))


def native_replay_problem(action, result):
    args = action.get("args") or {}
    if "codex_exit_code" in args and result.get("rc") != args["codex_exit_code"]:
        return "codex_exit_code_changed"
    path = args.get("pi_output_file")
    if path and (result.get("pi_output_file") != path or not result.get("pi_output_sha256")):
        return "pi_output_missing"
    if path:
        historical = args.get("pi_historical_output") or {}
        actual = dict(result.get("pi_output_dimensions") or {}, sha256=result.get("pi_output_sha256"))
        if any(actual.get(key) != value for key, value in historical.items()):
            return "pi_historical_output_mismatch"
    if args.get("pi_bash") and (args.get("pi_timeout_status") or result.get("termination_reproduced")):
        if (not args.get("pi_timeout_status") or not result.get("termination_reproduced")
                or result.get("timeout_s") != float(args["pi_timeout_status"])
                or result.get("rc") != 137 or result.get("fidelity") not in {"exact", "normalized"}):
            return "pi_timeout_changed"


def lenient_blocks(plan, report, rc, post_cut=None):
    """Blocking findings of the lenient policy: the three classes named in the module-level comment."""
    blocks = []
    if str(rc).strip() != "0":
        blocks.append({"rule": "replayer_failed", "detail": f"replay process exited rc={rc}"})
    results = list((report or {}).get("results") or [])
    actions = list((plan or {}).get("actions") or [])
    if len(results) != len(actions):
        blocks.append({"rule": "replay_incomplete",
                       "detail": f"{len(results)} of {len(actions)} planned actions replayed"})
    for action, result in zip(actions, results):
        cid = action.get("call_id")
        if result.get("call_id") != cid:
            blocks.append({"rule": "identity_mismatch", "call_id": cid})
            continue
        se, orig_ok = action.get("side_effect"), not action.get("is_error")
        if problem := native_replay_problem(action, result):
            blocks.append({"rule": problem, "call_id": cid})
            continue
        if recorded_no_match(action, result, lenient=True):
            result["fidelity"] = result["automatic_equivalence"] = "recorded_no_match"
            continue
        if harness_path_only(action, result, plan.get("harness_owned_paths", HARNESS_PATHS)):
            result["fidelity"] = result["automatic_equivalence"] = "harness_path_only"
            continue
        if se == "exec_background":
            orig_ok = action.get("bg_status") == "completed"
            if action.get("bg_status") == "failed":
                reasons = background_failure_reasons(action, result, results, actions=actions, lenient=True)
                if reasons:
                    blocks.append({"rule": "background_failure_changed", "call_id": cid, "conditions": reasons})
                else:
                    result["automatic_equivalence"] = result["fidelity"] = (
                        "bg_termination_reproduced" if action.get("bg_exit_code") in (124, 137, 143, 144)
                        else "bg_failure_reproduced")
        if result.get("pacing_unmet"):
            blocks.append({"rule": "pacing_unmet", "call_id": cid,
                           "detail": "original pacing of a running background job could not be kept"})
        if result.get("status") == "replayer_exception":
            blocks.append({"rule": "replayer_exception", "call_id": cid, "detail": str(result.get("error"))[:200]})
        if se in ("exec", "exec_background") and orig_ok:
            if result.get("status") == "timeout" and not result.get("orig_timed_out"):
                blocks.append({"rule": "replay_timeout", "call_id": cid,
                               "detail": "timed out where the original finished"})
            if result.get("rc") not in (None, 0) and not result.get("orig_timed_out"):
                blocks.append({"rule": "exit_code_changed", "call_id": cid, "rc": result.get("rc"),
                               "detail": "non-zero exit where the original succeeded"})
            if _violently_killed(action, result):
                blocks.append({"rule": "killed_or_oom", "call_id": cid, "rc": result.get("rc"),
                               "detail": "process killed (OOM/signal) where the original completed"})
        if se == "file_write" and orig_ok and result.get("status") not in ("applied", "skipped_orig_error"):
            # includes an unsupported write tool (status "skipped"): the original applied a change the replay did
            # not, so the file the agent works on afterwards differs
            blocks.append({"rule": "edit_not_applied", "call_id": cid, "status": result.get("status"),
                           "fidelity": result.get("fidelity"),
                           "detail": "the original edit applied; the replay could not apply it"})
        if se == "exec_background":
            if not action.get("bg_finished_before_k", True):
                blocks.append({"rule": "background_in_flight_at_cut", "call_id": cid,
                               "output_file": action.get("background_output_file"),
                               "detail": "background job still running at the cut; the agent uses its output later"})
            elif action.get("background_output_file") and not result.get("background_output_file"):
                blocks.append({"rule": "background_output_missing", "call_id": cid,
                               "path": action.get("background_output_file")})
    missing = list(post_cut.get("missing") or []) if isinstance(post_cut, dict) else []
    for path in missing:
        blocks.append({"rule": "post_cut_file_missing", "path": path,
                       "detail": "the agent reads this file after the cut; the replayed environment lacks it"})
    return blocks


def lenient_report(plan, report, post_cut=None, blocks=None):
    """Per-task divergence report: what differed (invisible to the agent) and what blocked (if anything)."""
    blocks = lenient_blocks(plan, report, (report or {}).get("rc", 0), post_cut) if blocks is None else list(blocks)
    entries, by_kind = [], {}
    for action, result in zip((plan or {}).get("actions") or [], (report or {}).get("results") or []):
        d = lenient_divergence(action, result)
        if d:
            entries.append(d)
            by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
    missing = list((post_cut or {}).get("missing") or []) if isinstance(post_cut, dict) else []
    return {"policy": "lenient", "n_actions": len((plan or {}).get("actions") or []),
            "n_divergent": len(entries), "by_kind": by_kind, "divergences": entries,
            "changed_numbers": [e["call_id"] for e in entries if e["kind"] == "changed_numbers"],
            "text_differs": [e["call_id"] for e in entries if e["kind"] == "text_differs"],
            "version_diffs": [{"call_id": e["call_id"], "versions": e.get("versions")}
                              for e in entries if e.get("versions")],
            "listing_or_line_order": [e["call_id"] for e in entries
                                      if e["kind"] == "listing_or_line_order"],
            "files_missing": missing,
            "post_cut_paths": (post_cut or {}).get("n") if isinstance(post_cut, dict) else 0,
            "blocks": blocks}


def validate_replay(plan, report, rc, reviewed_exceptions=None, policy="strict", post_cut=None):
    """Replay gate. policy="strict" keeps the historical fail-closed behaviour; "lenient" accepts output
    divergences and blocks only the three classes in lenient_blocks (see the module comment)."""
    if policy not in ("strict", "lenient"):
        raise RuntimeError(f"unknown replay policy {policy!r}")
    if policy == "lenient":
        blocks = lenient_blocks(plan, report, rc, post_cut)
        if blocks:
            raise RuntimeError("Lenient replay blocked: " + "; ".join(
                "/".join(str(b.get(k)) for k in ("rule", "call_id", "path", "detail") if b.get(k))
                for b in blocks))
        return lenient_report(plan, report, post_cut, blocks)
    import hashlib
    reviewed_exceptions = reviewed_exceptions or {}
    if str(rc).strip() != '0' or len(report.get('results', [])) != len(plan['actions']):
        raise RuntimeError('Replay did not complete all planned actions')
    allowed={'exact','normalized','error_reproduced','orig_error_not_applied'}
    # Line-order classes of the replayer are accepted only when the reordering is explained by stderr placement
    # (Claude Code interleaves stderr into the text differently); reordered stdout is never accepted silently.
    order_maps = {'same_lines_reordered': None, 'line_alignment_and_order_only': lambda l: l.strip()}
    import re as _re
    run_start = (plan.get('actions') or [{}])[0].get('timestamp')  # original run's first action (UTC)
    for result in report['results']:
        result['_run_start'] = run_start
    for action, result in zip(plan['actions'],report['results']):
        if result.get('call_id') != action['call_id']:
            raise RuntimeError('Replay action identity mismatch')
        if result.get('pacing_unmet'):
            raise RuntimeError('Replay could not keep the original pacing of a running background job: '
                               + action['call_id'])
        if problem := native_replay_problem(action, result):
            raise RuntimeError(f"{problem}: {action['call_id']}")
        if recorded_no_match(action, result):
            result["fidelity"] = result["automatic_equivalence"] = "recorded_no_match"
            continue
        if harness_path_only(action, result, plan.get("harness_owned_paths", HARNESS_PATHS)):
            result["fidelity"] = result["automatic_equivalence"] = "harness_path_only"
            continue
        fidelity=result.get('fidelity')
        exit_reviewed = False
        if action['side_effect']=='exec_background':
            # completed -> rc 0. failed -> the same non-zero exit code as the original task-notification reported
            # (Claude Code: "failed with exit code N"; unknown -> refused) AND a reviewed exception of kind
            # background_failure bound to the command and to the sha256 of the restored output file (the reviewer
            # compared it with the original failure; nothing is inferred automatically).
            orig_code = action.get('bg_exit_code')
            failed_ok = (action.get('bg_status') == 'failed' and orig_code not in (None, 0)
                         and result.get('rc') == orig_code
                         and reviewed_exceptions.get(action['call_id'], {}).get('kind') == 'background_failure'
                         and reviewed_exceptions[action['call_id']].get('command_sha256')
                         == hashlib.sha256((action.get('args') or {}).get('command', '').encode()).hexdigest()
                         and reviewed_exceptions[action['call_id']].get('background_output_sha256')
                         == result.get('background_output_sha256'))
            failed_ok = failed_ok or background_failure_equivalent(action, result, report["results"])
            status_same = (action.get('bg_status') == 'completed' and result.get('rc') == 0) or failed_ok
            if not (action.get('bg_finished_before_k',True) and status_same
                    and fidelity in ('bg_rerun_foreground', 'bg_termination_reproduced', 'bg_failure_reproduced')
                    and result.get('background_output_file')==action.get('background_output_file')
                    and result.get('background_output_sha256')):
                raise RuntimeError('Completed background output not restored: '+action['call_id'])
        elif (action['side_effect']=='task_mgmt' and action.get('tool') in ('TaskCreate','TaskUpdate')
              and fidelity=='task_mgmt_restored_from_session'):
            result['deferred_gate'] = 'session auxiliary files must be hash-verified after clone seeding, before CC starts'
        elif fidelity in order_maps and _stderr_placement_equal(action, result, order_maps[fidelity]):
            result['order_explained_by'] = 'stderr_placement'
        elif fidelity not in allowed and action['side_effect'] == 'exec' and automatic_equivalence(action, result):
            result['automatic_equivalence'] = automatic_equivalence(action, result)
        elif fidelity not in allowed:
            import hashlib, json, re
            review = reviewed_exceptions.get(action['call_id'], {})
            command = action.get('args', {}).get('command', '')
            last_line = (command.rstrip().splitlines() or [''])[-1]
            old = action.get('orig_text', '').strip().splitlines()
            new = result.get('stdout_full', '').strip().splitlines()
            version_line = lambda s: re.sub(r'\(main, [^)]+\)', '(main, <build-stamp>)', s)
            hardware_only = (review.get('kind') in ('hardware_probe','hardware_probe_and_python_build_stamp') and
                             review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                             len(old) >= 9 and len(new) >= 9 and old[1:7] == new[1:7] and
                             (old[0] == new[0] or (review.get('kind') == 'hardware_probe_and_python_build_stamp' and version_line(old[0]) == version_line(new[0]))) and
                             old[7].split() == new[7].split() and old[8].startswith('Mem:') and new[8].startswith('Mem:') and
                             not result.get('stderr_full') and result.get('rc') == 0)
            memory_pattern = r'(?m)^Mem:\s+(?:\d+\s+){5}\d+\s*$'
            memory_only = (review.get('kind') == 'shell_memory_probe' and
                           review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                           len(re.findall(memory_pattern, '\n'.join(old))) == 1 and
                           len(re.findall(memory_pattern, '\n'.join(new))) == 1 and
                           re.sub(memory_pattern, 'Mem:<host-memory>', '\n'.join(old)) ==
                           re.sub(memory_pattern, 'Mem:<host-memory>', '\n'.join(new)) and
                           not result.get('stderr_full') and result.get('rc') == 0)
            terminal_grep_only = (review.get('kind') == 'terminal_grep_no_match' and
                review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                bool(re.search(r'\|\s*grep\s', last_line)) and
                result.get('rc') == 1 and not action.get('is_error') and not result.get('stderr_full') and
                action.get('orig_text', '').strip() == result.get('stdout_full', '').strip())
            from .replay_inside import norm, original_timeout
            clock = re.compile(r'^(?:real|user|sys)\s+(?:\d+m\d+(?:\.\d+)?s|\d+(?:\.\d+)?)$')
            no_clock = lambda text: '\n'.join(line for line in norm(text).splitlines() if not clock.fullmatch(line)).strip()
            bound = review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest()
            timed_grep = (review.get('kind') == 'terminal_grep_with_shell_time' and bound and
                bool(re.search(r'\|\s*grep\s', last_line)) and
                result.get('rc') == 1 and not action.get('is_error') and
                all(clock.fullmatch(line) for line in result.get('stderr_full','').splitlines() if line.strip()) and
                no_clock(action.get('orig_text','')) == norm(result.get('stdout_full','')))
            old_to = original_timeout(action.get('orig_text'))
            timeout_same = (review.get('kind') == 'original_timeout_partial_output' and bound and old_to is not None and
                action.get('is_error') and result.get('orig_timed_out') and fidelity == 'timeout_reproduced' and
                result.get('status') == 'timeout' and result.get('timeout_s') == old_to and result.get('rc') is None and
                not result.get('stderr_full') and
                no_clock('\n'.join(action['orig_text'].splitlines()[2:])) == no_clock(result.get('stdout_full','')))
            profile_old = profile_without_times(action.get('orig_text', '')) if review.get('kind') in ('cprofile_timings','cprofile_reviewed_aggregate_counts') else None
            profile_new = profile_without_times(result.get('stdout_full','')) if profile_old is not None else None
            aggregate_reviewed = (profile_old is not None and profile_new is not None and
                review.get('kind') == 'cprofile_reviewed_aggregate_counts' and
                list(profile_old[1]) == review.get('original_aggregate_counts') and
                list(profile_new[1]) == review.get('replayed_aggregate_counts') and
                profile_old[0] == profile_new[0] and profile_old[2:] == profile_new[2:])
            profile_only = (profile_old is not None and
                review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest() and
                'cProfile.Profile()' in command and result.get('rc') == 0 and not result.get('stderr_full') and
                (profile_old == profile_new or aggregate_reviewed))
            # `ls -l[a]` of directories: some sandbox filesystems (image layers lazily mounted) omit the `.`/`..`
            # entries, which also changes the `total` block count. Every other row must be byte-identical.
            ls_dot = re.compile(r'^d\S+\s+\d+\s+\S+\s+\S+\s+\d+\s+\S+\s+\d+\s+\S+\s\.{1,2}$')
            # ls re-pads columns to the widest entry, so rows are compared as whitespace-split fields.
            no_dots = lambda lines: [l.split() for l in lines
                                     if not ls_dot.fullmatch(l) and not re.fullmatch(r'total \d+', l)]
            ls_dot_only = (review.get('kind') == 'ls_dot_entries' and bound and
                re.search(r'\bls\s+-\w*l', command) is not None and result.get('rc') == 0 and
                not result.get('stderr_full') and not action.get('is_error') and
                no_dots(old) == no_dots(new) and len(no_dots(old)) > 0)
            # Unseeded RNG in a read-only probe: same text with every number masked, explicitly reviewed.
            num_re = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
            unseeded = (review.get('kind') == 'unseeded_rng_numbers' and bound and fidelity == 'numbers_differ' and
                'random' in command and 'seed' not in command and result.get('rc') == 0 and
                not result.get('stderr_full') and not action.get('is_error') and
                num_re.sub('<n>', '\n'.join(old)) == num_re.sub('<n>', '\n'.join(new)))
            # CC showed only a <persisted-output> preview of a large output; the full original output is the bundled
            # tool-results file. Accept when the replayed stdout hashes to that file's reviewed sha256 and the
            # original preview is a prefix of the replayed stdout.
            prev = re.search(r'Preview \(first [^)]*\):\n(.*)', action.get('orig_text', ''), re.S)
            persisted_same = (review.get('kind') == 'persisted_output_identical' and bound and
                fidelity == 'persisted_preview' and result.get('rc') == 0 and not result.get('stderr_full') and
                not action.get('is_error') and prev is not None and
                hashlib.sha256(result.get('stdout_full', '').encode()).hexdigest() == review.get('persisted_output_sha256') and
                result.get('stdout_full', '').startswith(prev.group(1)[:1500]))
            # Read-only probe that picks glob.glob(...)[0] without sorting: the directory order of the sandbox
            # filesystem decides which file is inspected. Accept only command-bound, reviewed read-only probes.
            writes = re.compile(r"open\([^)]*['\"][wa]b?['\"]|\.to_(?:csv|parquet|pickle|json)\(|savefig|np\.save|"
                                r"\.write\(|mkdir|>\s*/|\brm\b|\bmv\b|\bcp\b|shutil|os\.remove|unlink")
            glob_order = (review.get('kind') == 'readonly_unsorted_glob_pick' and bound and
                fidelity in ('differs', 'numbers_differ') and re.search(r'glob\.glob\([^)]*\)\[0\]', command) is not None and
                writes.search(command) is None and result.get('rc') == 0 and not action.get('is_error'))
            # One unlabelled duration printed by a command that measures wall time (e.g. `print(fit(x), time.time()-t)`)
            num_re2 = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
            diff_nums = ([(x.group(), y.group()) for a_, b_ in zip(old, new) for x, y in
                          zip(num_re2.finditer(a_), num_re2.finditer(b_)) if x.group() != y.group()]
                         if len(old) == len(new) else None)
            timing_number = (review.get('kind') == 'unlabelled_timing_number' and bound and
                re.search(TIMING_CALL, command) is not None and diff_nums is not None and len(diff_nums) == 1 and
                num_re2.sub('#', '\n'.join(old)) == num_re2.sub('#', '\n'.join(new)) and
                bool(result.get('rc')) == bool(action.get('is_error')))
            # Same items listed in a different (filesystem) order by a command that also writes files: reviewed,
            # because the reviewer must check that the writes do not depend on that order.
            listing_line = (review.get('kind') == 'listing_order_line' and bound and len(old) == len(new) and
                all(sorted(re.findall(r"[\w.-]+", a_)) == sorted(re.findall(r"[\w.-]+", b_)) for a_, b_ in zip(old, new))
                and bool(result.get('rc')) == bool(action.get('is_error')))
            # The original read the agent's own session/log files (seeded only at launch, so absent during replay).
            # Reviewed: the reviewer confirms the command writes nothing; the replay must fail with "No such file"
            # on such a path and produce no stdout.
            agent_state = (review.get('kind') == 'agent_state_read' and bound and result.get('rc') not in (None, 0)
                and not result.get('stdout_full', '').strip()
                and re.search(r"No such file or directory:? '?(?:/logs/agent/|\S*\.claude/)", result.get('stderr_full', ''))
                is not None)
            # A WebFetch/WebSearch in the prefix: an external read the gate cannot (and must not) re-issue. Reviewed
            # as read-only: the tool input is unchanged (bound by sha256) and no local file is written by the action,
            # so the resumed agent's history keeps the original observation while the environment is untouched.
            web_readonly = (review.get('kind') == 'web_readonly_not_replayed' and
                fidelity == 'web_not_replayed' and
                review.get('args_sha256') == hashlib.sha256(
                    json.dumps(action.get('args') or {}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                and action.get('side_effect') == 'web')
            # The reviewer compared the two outputs line by line: they carry the same information once machine-
            # specific and volatile tokens are masked (see mask_volatile) and the command's exit outcome is
            # unchanged. Permitted only when the exception records that none of the changed numbers reappears in
            # the agent's later reasoning or tool arguments (changed_numbers_reused_later is False), the action ran
            # (not a timeout), and filesystem order is ignored only for a listing command.
            masked_only = (review.get('kind') == 'reviewed_masked_output' and bound and
                review.get('mask_version') == MASK_VERSION and review.get('order') in ('ordered', 'sorted', 'subsequence') and
                fidelity != 'timeout' and result.get('status') == 'ran' and
                bool(result.get('rc')) == bool(action.get('is_error')) and
                review.get('changed_numbers_reused_later') is False and
                (review.get('order') == 'ordered' or re.search(r"\b(?:ls|find|du|tree|stat|wc)\b", command)) and
                masked_equal(action.get('orig_text', ''), _replayed_text(result), review.get('order')))
            # Pattern-free review: a human compared this exact replay output with the original and approved it.
            # Bound to the command and to the sha256 of the replay's stdout + NUL + stderr + NUL + rc.
            exact_reviewed = (review.get('kind') == 'reviewed_replay_output' and bound and
                review.get('replay_output_sha256') == replay_output_sha256(result))
            # A Python `set` printed directly: the repr order follows the hash seed, so the same items can print in
            # another order. Accept only when every line is identical after canonicalising Python set literals of
            # quoted strings (items sorted inside the braces) and at least one line did differ.
            def _canon_sets(lines):
                """Sort the members of every Python `set` literal of strings, leaving everything else alone.

                Braces are located by a scanner that skips quoted strings (so a set printed *inside* a string is
                never rewritten) and the group is only accepted when `ast.literal_eval` returns a set of strings.
                """
                import ast

                def fix_line(line):
                    out, i, n = [], 0, len(line)
                    while i < n:
                        ch = line[i]
                        if ch in ("'", '"'):
                            j = i + 1
                            while j < n:
                                if line[j] == "\\":
                                    j += 2
                                    continue
                                if line[j] == ch:
                                    break
                                j += 1
                            out.append(line[i:j + 1])
                            i = j + 1
                            continue
                        if ch == '{':
                            depth, j = 0, i
                            while j < n:
                                if line[j] == '{':
                                    depth += 1
                                elif line[j] == '}':
                                    depth -= 1
                                    if depth == 0:
                                        break
                                elif line[j] in ("'", '"'):
                                    q = line[j]
                                    j += 1
                                    while j < n and line[j] != q:
                                        j += 2 if line[j] == "\\" else 1
                                j += 1
                            group = line[i:j + 1]
                            try:
                                val = ast.literal_eval(group)
                            except (ValueError, SyntaxError):
                                val = None
                            if isinstance(val, set) and val and all(isinstance(x, str) for x in val):
                                out.append('{' + ', '.join(repr(x) for x in sorted(val)) + '}')
                            else:
                                out.append(group)
                            i = j + 1
                            continue
                        out.append(ch)
                        i += 1
                    return ''.join(out)
                return '\n'.join(fix_line(l) for l in lines)
            set_order = (review.get('kind') == 'python_set_repr_order' and bound and fidelity == 'differs' and
                result.get('rc') == 0 and not result.get('stderr_full') and not action.get('is_error') and
                len(old) == len(new) and old != new and _canon_sets(old) == _canon_sets(new))
            if not (hardware_only or memory_only or terminal_grep_only or profile_only or timed_grep or timeout_same
                    or ls_dot_only or unseeded or persisted_same or glob_order or timing_number or listing_line
                    or agent_state or exact_reviewed or set_order or masked_only or web_readonly):
                raise RuntimeError('Unaccepted replay fidelity '+str(fidelity)+': '+action['call_id'])
            exit_reviewed = terminal_grep_only or timed_grep or timeout_same or agent_state or exact_reviewed
            result['reviewed_exception'] = ('exact replay output reviewed: ' + str(review.get('note', ''))[:200] if exact_reviewed else
                'reads the agent\'s own session/logs (seeded at launch, absent in replay); reviewed as read-only' if agent_state else
                'single unlabelled wall-time number differs (command measures time)' if timing_number else
                'same items listed in a different filesystem order; writes reviewed as order-independent' if listing_line else
                'replayed stdout identical to the persisted full original output (CC showed a preview)' if persisted_same else
                'read-only probe; unsorted glob(...)[0] picked a different file in this filesystem order' if glob_order else
                'identical ls rows; only the filesystem-dependent ./.. entries and total block count differ' if ls_dot_only else
                'unseeded RNG probe: identical text with numbers masked; reviewed as not affecting task state' if unseeded else
                'identical full stdout; command-bound terminal grep no-match returns1 while original CC reported non-error' if terminal_grep_only else
                'same original timeout deadline and full partial scientific output; shell timing differences only' if timeout_same else
                'identical scientific stdout; command-bound terminal grep returns1 and stderr contains only shell clocks' if timed_grep else
                'identical scientific output and profiled function identities/counts; explicitly bound aggregate profiler-count difference plus timings' if profile_only and aggregate_reviewed else
                'identical scientific output, cProfile call counts and profiled function identities; performance timings/order only' if profile_only else
                'python set(s) printed directly: identical items, hash-seed repr order only' if set_order else
                'external read not re-issued (WebFetch/WebSearch); read-only, no local state' if web_readonly else
                'reviewed: same output modulo machine spec, dates, addresses and numbers; no changed number reused later' if masked_only else
                'same package/version/CPU output; host-memory report only' if memory_only else
                'same Python release/compiler, scientific package versions and CPU count; reviewed Python build timestamp and host free-memory report differences')
        if action['side_effect']=='exec' and not exit_reviewed and bool(result.get('rc')) != bool(action.get('is_error')):
            raise RuntimeError('Replay exit outcome changed: '+action['call_id'])
    return True


def check_hashes(expected, actual, label):
    missing_or_different=[p for p,h in expected.items() if actual.get(p)!=h]
    if missing_or_different:
        raise RuntimeError(label+' hash mismatch: '+', '.join(missing_or_different))
    return True
