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
TIME_REPORT = re.compile(r"^(?:real|user|sys)\s+(?:\d+m\d+(?:\.\d+)?s|\d+(?:\.\d+)?)$")


DURATION = re.compile(r"(\d+(?:\.\d+)?)(\s?)(s|ms|sec|secs|seconds)\b")


def _timing_context(command, result):
    """The command's own code (inline, or a script it runs) measures wall time. The shell `time` keyword is not a
    timing context for printed lines: its report is handled separately (shell_time_report_only)."""
    return bool(re.search(TIMING_CALL, command) or result.get("scripts_measure_time"))


def _is_measurement(before, after):
    """Context of a duration value (number[+unit]) that shows it was measured, not configured: directly after a
    timing label (`took 0.13s`, `elapsed: 4`), in parentheses (`(0.07s)`), after `OK`, or at the end of a
    `<several words>: <duration>` line (`niter 6 dt_g None: 0.129s`). `key: 10 s`, `key=60s`, `timeout60s` are data."""
    return bool(re.search(TIMING_BEFORE, before)
                or (re.search(r"\(\s*$", before) and re.match(r"\s*\)", after))
                or re.search(r"\bOK\s+$", before)
                or (re.search(r"\S\s+\S[^=]*:\s*$", before) and re.fullmatch(r"\s*(?:s|ms|sec|secs|seconds)?\s*", after)))


def _durations_close(a, b):
    """Same line except measured durations that differ by at most 50% (run-to-run jitter). A differing duration
    counts as a measurement only in a measurement form: directly after a timing label (`took 0.13s`, `time: 4 s`),
    in parentheses (`(0.07s)`), after `OK`, or as a `<label>: <duration>` line end (`niter 6 dt_g None: 0.129s`)
    where the label is not a key (no `=`, and the line is not `key: value` with a single identifier key).
    Configuration-like values (`retry_interval=60s`, `exposure: 10 s`, `timeout60s`) are data."""
    if DURATION.sub("<D>", a) != DURATION.sub("<D>", b):
        return False
    for x, y in zip(DURATION.finditer(a), DURATION.finditer(b)):
        if x.group() == y.group():
            continue
        if not _is_measurement(a[:x.start()], a[x.end():]):
            return False
        u, v = float(x.group(1)), float(y.group(1))
        if x.group(3) != y.group(3) or abs(u - v) > max(0.5 * max(u, v), 0.05):
            return False
    return True


def _created_during_run(ls_date, run_start):
    """The original `ls -l` date of a row is at or after the original run's first action (UTC; the task image's
    clock), i.e. the agent created or modified that file during the run, so the replay necessarily re-created it at
    a new time. Task inputs (older mtimes, or the `Mon DD YYYY` form) must keep their date. Unknown -> False."""
    import datetime as dt
    if not run_start:
        return False
    m = re.fullmatch(r"([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2}):(\d{2})", ls_date.strip())
    if not m:
        return False
    try:
        t0 = dt.datetime.fromisoformat(run_start.replace("Z", "+00:00")).replace(tzinfo=None)
        mon = dt.datetime.strptime(m.group(1), "%b").month
        cand = dt.datetime(t0.year, mon, int(m.group(2)), int(m.group(3)), int(m.group(4)))
    except ValueError:
        return False
    if cand > t0 + dt.timedelta(days=200):  # ls shows the year only for old files: Dec listing in a Jan run
        cand = cand.replace(year=t0.year - 1)
    return cand >= t0 - dt.timedelta(minutes=1)  # minute resolution


def line_equivalent(a, b, command, result):
    """One output line vs its replay, with only evidence-bound differences: an `ls -l` row whose date column alone
    differs (command runs ls -l; name exact), or durations in a timing context within jitter."""
    if a == b:
        return True
    if re.search(r"\bls\s+-\w*l", command):
        ma, mb = LS_ROW.match(a), LS_ROW.match(b)
        if (ma and mb and ma.group(1).split() == mb.group(1).split() and ma.group(3) == mb.group(3)
                and _created_during_run(ma.group(2), result.get("_run_start"))
                and _created_during_run(mb.group(2), result.get("_run_start"))):
            return True
    return _timing_context(command, result) and _durations_close(a, b)


TIMING_BEFORE = r"(?i)\b(?:time|timing|elapsed|took|duration|runtime|wall(?:time)?|wall[ -]clock)\s*[:=]?\s*$"
TIMING_AFTER = r"(?i)^\s*(?:s|sec|secs|seconds|ms|min|minutes)\b"


def _only_timing_numbers_differ(a, b):
    """Same line with every differing number directly labelled as a duration ("time 0.85", "took 3.2 s")."""
    import re
    num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    ma, mb = list(num.finditer(a)), list(num.finditer(b))
    if len(ma) != len(mb) or num.sub("#", a) != num.sub("#", b):
        return False
    for x, y in zip(ma, mb):
        if x.group() == y.group():
            continue
        if not _is_measurement(a[:x.start()], a[x.end():]):
            return False
        u, v = abs(float(x.group())), abs(float(y.group()))
        if abs(u - v) > max(0.5 * max(u, v), 0.05):  # run-to-run jitter only
            return False
    return True
SHELL_TIME = r"(?:^|[;&|({]\s*)time\s|\bTIMEFORMAT=|/usr/bin/time\b"
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


def _shell_part(command):
    s = _heredoc_split(command)[0]
    return s.replace("2>&1", "").replace(">/dev/null", "").replace("> /dev/null", "").replace("2>/dev/null", "")


def _heredoc_part(command):
    return _heredoc_split(command)[1]


def _final_stage_is_grep(command):
    """The command's exit status is that of a final plain pipeline ending in grep/rg/egrep, so status 1 can only
    mean "no match". Shell text (heredoc bodies excluded) is split at newlines and `;` (the last command always runs
    and alone sets the status); that last command must be `a | b | grep ...` with no `&&`, `||`, `&` (a failing
    left side could return 1 itself), no stderr redirection (a failing earlier stage must show on stderr), no
    subshell/substitution/braces; `pipefail` anywhere disqualifies. Anything else is ambiguous."""
    sh = _heredoc_split(command)[0]
    # errexit (set -e / -o errexit / bash -e) can stop before the grep runs, with the same output
    if ("pipefail" in sh or "errexit" in sh or re.search(r"`|\$\(|\bset\s+-\w*e|\bbash\s+-\w*e", sh)
            or re.search(r"\bexit\b|\breturn\b|\btrap\b", sh)):
        return False
    final = [seg.strip() for seg in re.split(r"[\n;]", sh) if seg.strip()]
    if not final:
        return False
    last = final[-1]
    if re.search(r"[(){}]|&&|\|\||&|2>", last):
        return False
    stages = last.split("|")
    return len(stages) >= 2 and bool(re.match(r"(?:grep|egrep|rg)\s", stages[-1].strip()))


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
    different task state. Returns a reason string or None. Each class is narrow and checked on the full output:
      cc_grep_no_match      the command's last command (after the last newline/`;`) is a plain pipeline ending in
                            grep/rg (_final_stage_is_grep: no `&&`, `||`, `&`, stderr redirection, subshell; no
                            pipefail, errexit, exit/return/trap), replay rc=1 (no match), the original was not an error (CC renders that case as
                            success), the full stdout is identical; no stderr.
      ls_dot_entries_only   `ls -l*` listings whose only difference is the `.`/`..` rows and `total` block counts
                            (some sandbox filesystems omit them); every other row identical up to metadata-column
                            padding, file names exact.
      evidence_bound_volatile
                            line by line, in order, every differing line is an `ls -l` row whose date column alone
                            differs and whose original date is at/after the original run's first action (a file the
                            run created; command runs ls -l, file name exact), or, in a timing context (the command's
                            code or a script it runs measures time), the same line with durations in a measurement
                            form (_durations_close) within 50% of each other. No other masking.
      shell_diagnostic_prefix_only
                            bash's own error prefix differs (`/bin/bash: eval: line 18:` vs `bash: -c: line 9:`);
                            the original equals the replay's stdout in order once the replay's stderr lines are
                            taken out (stderr placement is the only other freedom).
      shell_time_report_only
                            the command uses the shell `time` keyword (TIMEFORMAT) or /usr/bin/time, or reads the
                            output file of an earlier background command that does (provenance from the plan);
                            after removing
                            the timing report lines and blank lines the output is equal line by line in order
                            (line_equivalent, as for evidence_bound_volatile).
      ls_mtime_and_stderr_placement_only
                            the command runs `ls -l`; only the date column of rows in ls -l format differs, for rows
                            whose original date is at/after the original run's first action (files the run created
                            and the replay re-created), plus the stderr placement freedom above; stdout order kept.
      cprofile_timings_only a cProfile/pstats table (possibly truncated by head) with identical rows: call counts and
                            function identities in the same order; only the four timing columns and the header's
                            total seconds differ. All non-table lines identical.
      timing_lines_only     the command measures wall time (time.time()/perf_counter/...), the exit outcome is the
                            same, and every differing number sits directly after a duration label ("time", "took",
                            "elapsed", ...) or before a time unit ("s", "ms", ...) and is within run-to-run jitter
                            (50%); everything else byte-identical after whitespace normalisation.
    """
    import re
    from .replay_inside import norm
    if result.get("status") != "ran" or result.get("timeout"):
        return None
    command = (action.get("args") or {}).get("command", "")
    # Claude Code reports a command whose last pipeline stage is grep/rg returning 1 (no match) as a non-error.
    if (result.get("rc") == 1 and not action.get("is_error") and _final_stage_is_grep(command)
            and not result.get("stderr_full")
            and norm(action.get("orig_text") or "") == norm(result.get("stdout_full", ""))):
        return "cc_grep_no_match"
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
    # `time cmd` / TIMEFORMAT / /usr/bin/time reports go to stderr, which Claude Code interleaves differently
    if re.search(SHELL_TIME, command, re.M) or result.get("_reads_time_report"):
        tfmt = re.search(r"TIMEFORMAT=['\"]?([^'\";]*)", command)
        rep = [re.compile(r"^(?:real|user|sys)\s+\d+m\d+(?:\.\d+)?s$|^(?:real|user|sys)\s+\d+(?:\.\d+)?$")]
        if tfmt:
            pat = re.escape(tfmt.group(1).strip())
            pat = re.sub(r"%\\?\d?l?[RUSP]", r"\\d+(?:\\.\\d+)?", pat.replace("\\%", "%"))
            rep.append(re.compile("^" + pat + "$"))
        drop = lambda ls: [l for l in ls if l.strip() and not any(r.match(l.strip()) for r in rep)]
        n_rep = sum(1 for l in old + new if l.strip() and any(r.match(l.strip()) for r in rep))
        d_old, d_new = drop(old), drop(new)
        if n_rep and len(d_old) == len(d_new) and all(line_equivalent(x, y, command, result)
                                                       for x, y in zip(d_old, d_new)):
            return "shell_time_report_only"
    # `ls -l` rows of files the command itself (re)created carry the replay's mtime: mask only the date column of
    # rows in ls -l format, in commands that run ls -l, combined with stderr placement
    if re.search(r"\bls\s+-\w*l", command):
        def mask(l):
            # only dates at/after the original run's start: rows of files the run created (both sides; the replay's
            # own dates are later still). An input file whose date changed stays unmasked on the original side.
            m = LS_ROW.match(l)
            return (LS_ROW.sub(r"\1 <MTIME> \3", l)
                    if m and _created_during_run(m.group(2), result.get("_run_start")) else l)
        if (_stderr_placement_equal(action, result, mask)
                and not _stderr_placement_equal(action, result, None)):
            return "ls_mtime_and_stderr_placement_only"
    if re.search(r"\bcProfile\b|\bprofile\.run\(|pstats", command) and len(old) == len(new):
        row = re.compile(r"\s*(\d+(?:/\d+)?)\s+(?:\d+\.\d+\s+){4}(.+)")
        hdr = re.compile(r"(\s*\d+ function calls(?: \(\d+ primitive calls\))? in )\d+(?:\.\d+)?( seconds)")
        key = lambda l: (row.fullmatch(l).groups() if row.fullmatch(l) else ("", hdr.sub(r"\1<t>\2", l)))
        if any(row.fullmatch(l) for l in old) and [key(l) for l in old] == [key(l) for l in new]:
            return "cprofile_timings_only"
    if len(old) == len(new) and old != new and all(line_equivalent(x, y, command, result)
                                                     for x, y in zip(old, new)):
        return "evidence_bound_volatile"
    if re.search(TIMING_CALL, command) and len(old) == len(new):
        diff = [(a, b) for a, b in zip(old, new) if a != b]
        if diff and all(_only_timing_numbers_differ(a, b) for a, b in diff):
            return "timing_lines_only"
    return None


def validate_replay(plan, report, rc, reviewed_exceptions=None):
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
    planned_ids = {a['call_id'] for a in plan['actions']}
    # background output files written by a command that uses the shell `time` keyword: a later read of such a file
    # shows that command's timing report (provenance from the plan, not from the reading command)
    time_files = []
    for action, result in zip(plan['actions'], report['results']):
        f = action.get('background_output_file')
        if (f and action['side_effect'] == 'exec_background'
                and _re.search(SHELL_TIME, (action.get('args') or {}).get('command', ''), _re.M)):
            time_files.append(f)
        result['_run_start'] = run_start
        cmd = (action.get('args') or {}).get('command', '')
        result['_reads_time_report'] = any(tf in cmd for tf in time_files)
    for action, result in zip(plan['actions'],report['results']):
        if result.get('call_id') != action['call_id']:
            raise RuntimeError('Replay action identity mismatch')
        if result.get('pacing_unmet'):
            raise RuntimeError('Replay could not keep the original pacing of a running background job: '
                               + action['call_id'])
        fidelity=result.get('fidelity')
        exit_reviewed = False
        if action['side_effect']=='exec_background':
            # completed -> rc 0; failed (Claude Code: non-zero exit) -> non-zero rc, same as the original task status
            # completed -> rc 0. failed -> the same non-zero exit code as the original task-notification reported
            # (Claude Code: "failed with exit code N"; unknown -> refused) AND evidence that it failed the same way:
            # a later read of its output file that is part of this replay plan (that read compares the original
            # text with the restored file like any other read; bundles built with --include-read-only-reads keep
            # such reads), or a command-bound reviewed exception of kind background_failure.
            orig_code = action.get('bg_exit_code')
            failed_ok = (action.get('bg_status') == 'failed' and orig_code not in (None, 0)
                         and result.get('rc') == orig_code
                         and (any(c in planned_ids for c in action.get('bg_output_read_later') or ())
                              or reviewed_exceptions.get(action['call_id'], {}).get('kind') == 'background_failure'
                              and reviewed_exceptions[action['call_id']].get('command_sha256')
                              == hashlib.sha256((action.get('args') or {}).get('command', '').encode()).hexdigest()))
            status_same = (action.get('bg_status') == 'completed' and result.get('rc') == 0) or failed_ok
            if not (action.get('bg_finished_before_k',True) and status_same
                    and fidelity=='bg_rerun_foreground'
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
            exit_reviewed = result['automatic_equivalence'] == 'cc_grep_no_match'
        elif fidelity not in allowed:
            import hashlib, re
            review = reviewed_exceptions.get(action['call_id'], {})
            command = action.get('args', {}).get('command', '')
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
                bool(re.search(r'\|\s*grep\s', command.rstrip().splitlines()[-1])) and
                result.get('rc') == 1 and not action.get('is_error') and not result.get('stderr_full') and
                action.get('orig_text', '').strip() == result.get('stdout_full', '').strip())
            from .replay_inside import norm, original_timeout
            clock = re.compile(r'^(?:real|user|sys)\s+(?:\d+m\d+(?:\.\d+)?s|\d+(?:\.\d+)?)$')
            no_clock = lambda text: '\n'.join(line for line in norm(text).splitlines() if not clock.fullmatch(line)).strip()
            bound = review.get('command_sha256') == hashlib.sha256(command.encode()).hexdigest()
            timed_grep = (review.get('kind') == 'terminal_grep_with_shell_time' and bound and
                bool(re.search(r'\|\s*grep\s', command.rstrip().splitlines()[-1])) and
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
            if not (hardware_only or memory_only or terminal_grep_only or profile_only or timed_grep or timeout_same
                    or ls_dot_only or unseeded or persisted_same or glob_order or timing_number or listing_line
                    or agent_state):
                raise RuntimeError('Unaccepted replay fidelity '+str(fidelity)+': '+action['call_id'])
            exit_reviewed = terminal_grep_only or timed_grep or timeout_same or agent_state
            result['reviewed_exception'] = ('reads the agent\'s own session/logs (seeded at launch, absent in replay); reviewed as read-only' if agent_state else
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
