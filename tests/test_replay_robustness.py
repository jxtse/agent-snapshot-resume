"""Replay robustness for long prefixes: review-free equivalence classes of the replay gate, replayer pacing of shell
background jobs, background-output trailer, empty-output rendering, failed background tasks, and the offline screen."""
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from agent_snapshot_resume import screen
from agent_snapshot_resume.replay import gate
from agent_snapshot_resume.replay import replay_inside as replay


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def one(command, orig, stdout, stderr="", rc=0, is_error=False, fidelity=None, side_effect="exec", **extra):
    a = {"call_id": "c1", "side_effect": side_effect, "tool": "Bash", "args": {"command": command},
         "orig_text": orig, "is_error": is_error, "timestamp": "2026-11-02T20:00:00.000Z", **extra}
    if fidelity is None:  # classify exactly as the replayer does
        fidelity = replay.compare(orig, replay.cc_bash_text({"stdout": stdout, "stderr": stderr}), rc)
    r = {"call_id": "c1", "fidelity": fidelity, "stdout_full": stdout, "stderr_full": stderr, "rc": rc,
         "status": "ran", "timeout": False}
    return {"actions": [a]}, {"results": [r]}


def passes(plan, report, reviewed=None):
    try:
        return gate.validate_replay(plan, report, 0, reviewed or {})
    except RuntimeError:
        return False


LS_OLD = "total 8\n-rw-r--r-- 1 root root 5 Oct 1 00:00 a.csv\n-rw-r--r-- 1 root root 7 Oct 1 00:00 b.csv"
LS_NEW = ("total 16\ndrwxr-xr-x 2 root root 4096 Oct 1 00:00 .\ndrwxr-xr-x 3 root root 4096 Oct 1 00:00 ..\n"
          "-rw-r--r-- 1 root root 5 Oct 1 00:00 a.csv\n-rw-r--r-- 1 root root 7 Oct 1 00:00 b.csv")


class BackgroundOutputOrder(unittest.TestCase):
    def test_merged_stream_keeps_write_order(self):
        # Claude Code writes a background task's stdout and stderr into one file in write order
        cmd = "echo a; echo b >&2; echo c"
        r = replay.run_bash(cmd, None, 10, merge_stderr=True)
        self.assertEqual((r["stdout"], r["stderr"], r["rc"]), ("a\nb\nc\n", "", 0))
        r = replay.run_bash(cmd, None, 10)
        self.assertEqual((r["stdout"], r["stderr"]), ("a\nc\n", "b\n"))


class TimeReportProvenance(unittest.TestCase):
    def plan(self, bg_cmd, read_cmd):
        f = "/tmp/claude-0/-root/s/tasks/b1.output"
        bg = {"call_id": "b1", "side_effect": "exec_background", "tool": "Bash", "args": {"command": bg_cmd},
              "orig_text": "Command running in background", "is_error": False, "background_output_file": f,
              "bg_status": "completed", "timestamp": "2026-11-02T20:00:00.000Z"}
        rd = {"call_id": "r1", "side_effect": "exec", "tool": "Bash", "args": {"command": read_cmd.format(f=f)},
              "orig_text": "=== t ===\nreal\t0m29.375s\nexit=0", "is_error": False}
        res_bg = {"call_id": "b1", "fidelity": "bg_rerun_foreground", "rc": 0, "status": "ran",
                  "background_output_file": f, "background_output_sha256": "x"}
        out = "=== t ===\nreal\t0m29.363s\nexit=0\n"
        res_rd = {"call_id": "r1", "fidelity": replay.compare(rd["orig_text"], out, 0), "stdout_full": out,
                  "stderr_full": "", "rc": 0, "status": "ran", "timeout": False}
        return {"actions": [bg, rd]}, {"results": [res_bg, res_rd]}

    def test_read_of_timed_background_output(self):
        self.assertFalse(passes(*self.plan("time python3 s.py; echo exit=$?", "echo '=== t ==='; grep -E 'real|exit' {f}")))

    def test_untimed_background_output_values_not_masked(self):
        self.assertFalse(passes(*self.plan("python3 s.py; echo exit=$?", "echo '=== t ==='; grep -E 'real|exit' {f}")))


class AutomaticEquivalence(unittest.TestCase):
    def test_ls_dot_entries(self):
        self.assertTrue(passes(*one("ls -la /app/data", LS_OLD, LS_NEW)))
        changed = LS_NEW.replace("7 Oct", "8 Oct")
        self.assertFalse(passes(*one("ls -la /app/data", LS_OLD, changed)))
        self.assertFalse(passes(*one("cat x", LS_OLD, LS_NEW)), "only for ls -l listings")

    def test_timing_lines(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('loss 0.25')\nprint('time', time.time()-t)\nEOF"
        self.assertFalse(passes(*one(cmd, "loss 0.25\ntime 0.851", "loss 0.25\ntime 0.862")))
        self.assertFalse(passes(*one(cmd, "loss 0.25\ntime 0.851", "loss 0.26\ntime 0.862")), "science number")
        self.assertFalse(passes(*one(cmd, "per-time err 0.1", "per-time err 0.2")), "label is not a duration")
        self.assertFalse(passes(*one("python3 -c 'print(1)'", "time 0.851", "time 0.862")), "command does not time")

    def test_exit_outcome_must_match(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('time', time.time()-t)\nEOF"
        p, r = one(cmd, "time 0.851", "time 0.862", rc=1)
        self.assertFalse(passes(p, r))

    def test_grep_no_match_rc1(self):
        cmd = "python3 run.py\npip list | grep -i pypdf"
        self.assertFalse(passes(*one(cmd, "saved 22", "saved 22\n", rc=1)))
        self.assertFalse(passes(*one(cmd, "saved 22", "saved 23\n", rc=1)))
        self.assertFalse(passes(*one("python3 run.py", "saved 22", "saved 22\n", rc=1)), "not a grep pipeline")
        for bad in ("echo done; printf x | grep x >/dev/null; test -f /app/required",
                    "echo done; test -f /app/required && cat f | grep x",
                    "echo done; cat f | grep x || true", "set -o pipefail; cat f | grep x",
                    "echo done; cat f 2>/dev/null | grep x", "echo done; (cat f | grep x)",
                    'set -e; printf "ready\\n"; test "$READY" = yes; printf "" | grep optional',
                    'set -o errexit; echo done; false; cat f | grep x', "bash -e -c 'echo done; false'; cat f | grep x",
                    "echo done; test -f x || exit 1; cat f | grep x"):
            self.assertFalse(passes(*one(bad, "done", "done\n", rc=1)), bad)

    def test_listing_order_is_not_automatic(self):
        # token-sorted comparison would hide swapped field values (review counterexample)
        cmd = "python3 - <<'EOF'\nimport glob\nfor d in glob.glob('/d/*'):\n    print(d, end=' ')\nEOF"
        self.assertFalse(passes(*one(cmd, "/d/a /d/b /d/c", "/d/b /d/a /d/c")))
        self.assertFalse(passes(*one("python3 x.py  # os.listdir(", "winner a loser b", "winner b loser a")))

    def test_shell_diagnostic_prefix(self):
        o = "Exit code 2\nout 1\n/bin/bash: eval: line 18: unexpected EOF while looking for matching `\"'"
        n = "out 1\nbash: -c: line 9: unexpected EOF while looking for matching `\"'"
        self.assertTrue(passes(*one('python3 -c "x', o, "out 1\n", stderr=n.split("\n")[1], rc=2, is_error=True)))
        self.assertFalse(passes(*one('python3 -c "x', o, "out 2\n", stderr=n.split("\n")[1], rc=2, is_error=True)))

    def test_shell_time_report(self):
        cmd = "TIMEFORMAT='%3R s'; time python3 solver.py; ls out"
        self.assertFalse(passes(*one(cmd, "6.098 s\nfit ok\na.json", "fit ok\na.json\n", stderr="6.103 s\n")))
        self.assertFalse(passes(*one(cmd, "6.098 s\nfit ok\na.json", "fit bad\na.json\n", stderr="6.103 s\n")))
        cmd2 = "time python3 solver.py"
        self.assertFalse(passes(*one(cmd2, "real\t0m6.7s\nuser\t0m6.8s\nsys\t0m0.4s\nok", "ok\n",
                                    stderr="\nreal\t0m6.9s\nuser\t0m6.9s\nsys\t0m0.5s\n")))

    def test_cprofile_timings(self):
        cmd = "python3 -m cProfile -s cumtime solver.py | head -8"
        o = ("  981 function calls (979 primitive calls) in 38.063 seconds\n\n   Ordered by: cumulative time\n\n"
             "   ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n"
             "    704/1    0.006    0.000   38.102   38.102 {built-in method builtins.exec}")
        n = o.replace("38.063", "38.886").replace("38.102   38.102", "38.925   38.925")
        self.assertFalse(passes(*one(cmd, o, n)))
        self.assertFalse(passes(*one(cmd, o, n.replace("704/1", "705/1"))), "call counts differ")

    def test_agent_state_read_needs_review(self):
        cmd = "python3 - <<'EOF'\nimport json\nfor l in open('/logs/agent/sessions/projects/-app/x.jsonl'): print(l)\nEOF"
        err = "FileNotFoundError: [Errno 2] No such file or directory: '/logs/agent/sessions/projects/-app/x.jsonl'\n"
        self.assertFalse(passes(*one(cmd, "Reconstruct the ...", "", stderr=err, rc=1)), "never automatic")
        rv = {"c1": {"kind": "agent_state_read", "command_sha256": sha(cmd)}}
        self.assertTrue(passes(*one(cmd, "Reconstruct the ...", "", stderr=err, rc=1), rv))
        self.assertFalse(passes(*one(cmd, "Reconstruct the ...", "partial", stderr=err, rc=1), rv), "stdout")
        touch = "touch /app/changed; " + cmd
        self.assertFalse(passes(*one(touch, "Reconstruct the ...", "", stderr=err, rc=1), rv), "bound to command")

    def test_volatile_with_stderr_placement(self):
        o = "<stdin>:15: RuntimeWarning: x\ndrwxr-xr-x 2 root root 4096 Nov  2 20:17 .\nres 1"
        n = "drwxr-xr-x 2 root root 4096 Nov 13 04:47 .\nres 1\n"
        self.assertFalse(passes(*one("python3 a.py; ls -la d", o, n, stderr="<stdin>:15: RuntimeWarning: x\n")))

    def test_reordered_stdout_rejected_reordered_stderr_accepted(self):
        self.assertFalse(passes(*one("python3 rank.py", "first a\nsecond b", "second b\nfirst a\n")))
        self.assertTrue(passes(*one("python3 rank.py", "warn x\nfirst a\nsecond b", "first a\nsecond b\n",
                                    stderr="warn x\n")))

    def test_bare_in_is_not_a_duration(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('samples in', n)\nEOF"
        self.assertFalse(passes(*one(cmd, "samples in 10", "samples in 11\n")))

    def test_full_timestamps_with_stderr_move_rejected(self):
        self.assertFalse(passes(*one("cat records", "warn\nevent 2020-01-01T00:00:00 a",
                                     "event 2021-05-05T00:00:00 a\n", stderr="warn\n")))

    def test_ls_mtime_only_in_ls_rows(self):
        # run started 2026-11-02T20:00Z (first action): a.csv dated 20:17 was created by the run
        o = "w\n-rw-r--r-- 1 root root 5 Nov  2 20:17 a.csv"
        n = "-rw-r--r-- 1 root root 5 Nov 13 04:47 a.csv\n"
        self.assertFalse(passes(*one("python3 mk.py; ls -la out", o, n, stderr="w\n")))
        self.assertFalse(passes(*one("python3 mk.py; ls -la out", o, n.replace(" 5 ", " 6 "), stderr="w\n")))
        self.assertFalse(passes(*one("cat log", "w\nrun Nov  2 20:17 ok", "run Nov 13 04:47 ok\n", stderr="w\n")))

    def test_input_file_mtime_change_rejected(self):
        # input.csv dated before the run started: its date must be identical (freshness checks depend on it)
        o = "-rw-r--r-- 1 root root 5 Oct  1 00:00 input.csv"
        n = "-rw-r--r-- 1 root root 5 Nov  1 00:00 input.csv\n"
        self.assertFalse(passes(*one("ls -l input.csv", o, n)))
        o2 = "total 8\n-rw-r--r-- 1 root root 5 Oct  1 00:00 input.csv"
        n2 = ("total 8\ndrwxr-xr-x 2 root root 4096 Nov 13 04:47 .\n"
              "-rw-r--r-- 1 root root 5 Nov  1 00:00 input.csv\n")
        self.assertFalse(passes(*one("ls -la d", o2, n2)))
        self.assertFalse(passes(*one("ls -l input.csv; make", o + "\nup to date", n + "up to date\n")))

    def test_task_timestamps_rejected_even_in_order_and_under_time(self):
        o, n = "event 2020-01-01T00:00:00 a", "event 2021-05-05T00:00:00 a\n"
        self.assertFalse(passes(*one("cat records", o, n)))
        self.assertFalse(passes(*one("time cat records", "real\t0m0.1s\n" + o, n, stderr="real\t0m0.2s\n")))
        # clock-read timestamps are not masked automatically either (a reviewed exception is needed)
        cmd = "python3 -c 'import datetime; print(datetime.datetime.now())'"
        self.assertFalse(passes(*one(cmd, "2026-11-02 20:17:01.5", "2026-11-13 04:47:09.1\n")))

    def test_ls_file_names_kept_exact(self):
        o = "total 8\n-rw-r--r-- 1 root root 5 Oct 1 00:00 a b"
        n = ("total 16\ndrwxr-xr-x 2 root root 4096 Oct 1 00:00 .\n"
             "-rw-r--r-- 1 root root 5 Oct 1 00:00 a  b")
        self.assertFalse(passes(*one("ls -la d", o, n)))

    def test_hex_and_data_never_masked(self):
        self.assertFalse(passes(*one("cat checksum.txt", "0x12345678", "0x87654321\n")))
        self.assertFalse(passes(*one("time cat m.txt", "real\t0m0.1s\nexposure 10 seconds",
                                     "exposure 90 seconds\n", stderr="real\t0m0.2s\n")))
        o = "total 8\n-rw-r--r-- 1 root root 5 Oct 1 00:00 a0x12345678"
        n = "total 8\ndrwxr-xr-x 2 root root 4096 Oct 1 00:00 .\n-rw-r--r-- 1 root root 5 Oct 1 00:00 a0x87654321"
        self.assertFalse(passes(*one("ls -la d", o, n)))

    def test_braced_shell_time_report(self):
        cmd = "{ time python3 -B solver.py ; } 2>&1 | tail -4"
        self.assertFalse(passes(*one(cmd, "done\nreal\t0m6.174s\nuser\t0m6.263s\nsys\t0m0.453s",
                                    "done\nreal\t0m6.082s\nuser\t0m6.169s\nsys\t0m0.461s\n")))

    def test_hex_with_stderr_move_rejected(self):
        self.assertFalse(passes(*one("python3 sum.py", "warning\nchecksum=0xabcdef", "checksum=0x123456\n",
                                     stderr="warning\n")))

    def test_duration_valued_data_rejected(self):
        self.assertFalse(passes(*one("time cat config.txt", "real\t0m0.1s\nretry_interval=60s",
                                     "retry_interval=90s\n", stderr="real\t0m0.2s\n")))
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint(open('c').read())\nEOF"
        self.assertFalse(passes(*one(cmd, "retry_interval=60s", "retry_interval=70s\n")))
        self.assertFalse(passes(*one(cmd, "timeout60s", "timeout70s\n")))
        self.assertFalse(passes(*one(cmd, "exposure: 10 s", "exposure: 15 s\n")), "key: value config")
        self.assertFalse(passes(*one("time python3 show_config.py", "exposure: 10 s", "exposure: 15 s\n")))
        p_, r_ = one("python3 show_config.py", "exposure: 10 s", "exposure: 15 s\n")
        r_["results"][0]["scripts_measure_time"] = True
        self.assertFalse(passes(p_, r_), "a script that measures time can still print config durations")

    def test_measurement_forms_accepted(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nEOF"
        for o, n in (("tau 20->21.55 (0.07s)", "tau 20->21.55 (0.08s)\n"),
                     ("niter 6 dt_g None: 0.129s", "niter 6 dt_g None: 0.130s\n"),
                     ("short40   OK 4.3s nout=10", "short40   OK 4.2s nout=10\n"),
                     ("fit took 3.2 s", "fit took 3.4 s\n")):
            self.assertFalse(passes(*one(cmd, o, n)), o)

    def test_durations_in_timing_context_within_jitter(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('OK %.1fs n=49' % (time.time()-t))\nEOF"
        self.assertFalse(passes(*one(cmd, "OK 6.0s n=49", "OK 6.1s n=49\n")))
        self.assertFalse(passes(*one(cmd, "OK 6.0s n=49", "OK 60.0s n=49\n")), "beyond jitter")
        self.assertFalse(passes(*one(cmd, "OK 6.0s n=49", "OK 6.1s n=50\n")), "other number")

    def test_script_timing_recorded_by_replayer(self):
        p, r = one("python3 gencheck.py", "OK 6.0s n=49", "OK 6.1s n=49\n")
        self.assertFalse(passes(p, r))
        r["results"][0]["scripts_measure_time"] = True
        self.assertFalse(passes(p, r))
        with tempfile.TemporaryDirectory() as td:
            Path(td, "g.py").write_text("import time\nt=time.time()\n")
            self.assertTrue(replay.scripts_measure_time("python3 -B g.py x", td))
            self.assertFalse(replay.scripts_measure_time("python3 other.py", td))

    def test_task_data_dates_reordered_are_rejected(self):
        # review counterexample: dates that are task data, changed and reordered, must not pass
        o = "event 2020-01-01 a\nevent 2020-01-02 b"
        n = "event 2021-05-05 b\nevent 2021-05-04 a\n"
        self.assertFalse(passes(*one("cat records", o, n)))


class ReviewedKinds(unittest.TestCase):
    def test_unlabelled_timing_number(self):
        cmd = "python3 -c \"import time\nt=time.time(); print(fit(), time.time()-t)\""
        o, n = "(0.0025, 53.1) 3.717", "(0.0025, 53.1) 3.713"
        rv = {"c1": {"kind": "unlabelled_timing_number", "command_sha256": sha(cmd)}}
        self.assertFalse(passes(*one(cmd, o, n, fidelity="numbers_differ")))
        self.assertTrue(passes(*one(cmd, o, n, fidelity="numbers_differ"), rv))
        self.assertFalse(passes(*one(cmd, o, "(0.0026, 53.1) 3.713", fidelity="numbers_differ"), rv), "two numbers")

    def test_listing_order_line(self):
        cmd = "python3 mksyn.py"
        rv = {"c1": {"kind": "listing_order_line", "command_sha256": sha(cmd)}}
        o, n = "made ['a', 'b', 'c']", "made ['b', 'c', 'a']"
        self.assertTrue(passes(*one(cmd, o, n), rv))
        self.assertFalse(passes(*one(cmd, o, "made ['b', 'c', 'd']"), rv))
        self.assertFalse(passes(*one(cmd + " ", o, n), rv), "review bound to the exact command")


class BackgroundTasks(unittest.TestCase):
    def bg(self, status, rc, code=None, read_later=(), review=None):
        a = {"call_id": "c1", "side_effect": "exec_background", "tool": "Bash", "args": {"command": "x"},
             "bg_finished_before_k": True, "bg_status": status, "background_output_file": "/tmp/t.output",
             "bg_exit_code": code, "bg_output_read_later": read_later}
        r = {"call_id": "c1", "fidelity": "bg_rerun_foreground", "rc": rc, "background_output_file": "/tmp/t.output",
             "background_output_sha256": "ab"}
        return {"actions": [a]}, {"results": [r]}, review

    def ok(self, *args, **kw):
        p, r, review = self.bg(*args, **kw)
        return passes(p, r, review)

    def test_completed(self):
        self.assertTrue(self.ok("completed", 0))
        self.assertFalse(self.ok("completed", 1))

    def test_failed_needs_same_code_and_output_evidence(self):
        sha = hashlib.sha256(b"x").hexdigest()
        rv = {"c1": {"kind": "background_failure", "command_sha256": sha, "background_output_sha256": "ab"}}
        self.assertTrue(self.ok("failed", 1, code=1, review=rv))
        self.assertFalse(self.ok("failed", 0, code=1, review=rv))
        self.assertFalse(self.ok("failed", 2, code=1, review=rv), "different exit code")
        self.assertFalse(self.ok("failed", 1, code=None, review=rv), "original exit code unknown")
        self.assertFalse(self.ok("failed", 1, code=1), "no evidence of the failure output")
        self.assertTrue(self.ok("failed", 1, code=1, review={"c1": {"kind": "background_failure", "command_sha256": sha,
                                                                    "background_output_sha256": "ab"}}))
        self.assertFalse(self.ok("failed", 1, code=1, review={"c1": {"kind": "background_failure", "command_sha256": "0"}}))


class RemovedAutomaticClassesNeedReview(unittest.TestCase):
    """Timing/grep/mtime/profile/listing/background-failure differences are never accepted automatically; they pass
    only with a command-sha256-bound reviewed exception."""
    def sha(self, cmd):
        return hashlib.sha256(cmd.encode()).hexdigest()

    def test_grep_no_match_via_reviewed_exception(self):
        cmd = "python3 run.py\npip list | grep -i pypdf"
        p_, r_ = one(cmd, "saved 22", "saved 22\n", rc=1)
        self.assertFalse(passes(p_, r_))
        self.assertTrue(passes(p_, r_, {"c1": {"kind": "terminal_grep_no_match", "command_sha256": self.sha(cmd)}}))
        self.assertFalse(passes(p_, r_, {"c1": {"kind": "terminal_grep_no_match", "command_sha256": "0" * 64}}))

    def test_timing_number_via_reviewed_exception(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nEOF"
        p_, r_ = one(cmd, "loss 0.25 0.851", "loss 0.25 0.862\n")
        self.assertFalse(passes(p_, r_))
        self.assertTrue(passes(p_, r_, {"c1": {"kind": "unlabelled_timing_number", "command_sha256": self.sha(cmd)}}))

    def test_background_failure_only_reviewed(self):
        a = {"call_id": "c1", "side_effect": "exec_background", "tool": "Bash", "args": {"command": "x"},
             "bg_finished_before_k": True, "bg_status": "failed", "background_output_file": "/tmp/t.output",
             "bg_exit_code": 1, "bg_output_read_later": ["r1"]}
        rd = {"call_id": "r1", "side_effect": "exec", "tool": "Bash", "args": {"command": "cat /tmp/t.output"},
              "orig_text": "Traceback: KeyError", "is_error": False}
        r = {"call_id": "c1", "fidelity": "bg_rerun_foreground", "rc": 1, "background_output_file": "/tmp/t.output",
             "background_output_sha256": "ab"}
        good = {"call_id": "r1", "fidelity": "exact", "rc": 0, "status": "ran", "stdout_full": "Traceback: KeyError\n"}
        self.assertFalse(passes({"actions": [a, rd]}, {"results": [r, good]}), "no inference from later reads")
        rv = {"c1": {"kind": "background_failure", "command_sha256": self.sha("x"), "background_output_sha256": "ab"}}
        self.assertTrue(passes({"actions": [a, rd]}, {"results": [r, good]}, rv))
        rv["c1"]["background_output_sha256"] = "cd"
        self.assertFalse(passes({"actions": [a, rd]}, {"results": [r, good]}, rv), "other failure output")

    def test_reviewed_replay_output_binds_exact_output(self):
        cmd = "time python3 solver.py"
        p_, r_ = one(cmd, "real\t0m6.7s\nok", "ok\n", stderr="real\t0m6.9s\n")
        self.assertFalse(passes(p_, r_))
        rv = {"c1": {"kind": "reviewed_replay_output", "command_sha256": self.sha(cmd),
                     "replay_output_sha256": gate.replay_output_sha256(r_["results"][0])}}
        self.assertTrue(passes(p_, r_, rv))
        r_["results"][0]["stdout_full"] = "ok2\n"
        self.assertFalse(passes(p_, r_, rv))


class CProfileSorted(unittest.TestCase):
    def test_time_sorted_rows_may_swap(self):
        hdr = "  9810144 function calls (9798384 primitive calls) in {} seconds"
        rows = ["     93    0.006    0.000    0.900    0.010 __init__.py:1(<module>)",
                "  37345    0.300    0.000    0.890    0.000 solver.py:504(add_point)"]
        o = "\n".join([hdr.format("38.063")] + rows)
        n = "\n".join([hdr.format("38.974")] + rows[::-1]) + "\n"
        self.assertFalse(passes(*one("python3 -m cProfile -s cumtime s.py | head", o, n)))
        self.assertFalse(passes(*one("python3 -m cProfile s.py | head", o, n)), "not time-sorted")
        # program output around the table keeps its order
        o3 = "case_A 1\ncase_B 2\n" + o
        n3 = "case_B 2\ncase_A 1\n" + n
        self.assertFalse(passes(*one("python3 -m cProfile -s cumtime s.py | head", o3, n3)), "stdout reordered")
        n2 = n.replace("37345", "37346")
        self.assertFalse(passes(*one("python3 -m cProfile -s cumtime s.py | head", o, n2)), "call count changed")


class FailedBackgroundEvidence(unittest.TestCase):
    def test_plan_records_exit_code_and_adds_output_read(self):
        from agent_snapshot_resume.trace.ir import replay_plan
        f = "/tmp/claude-0/-app/s/tasks/b1.output"
        tc = lambda cid, cmd, se, ts, **kw: {"type": "tool_call", "call_id": cid, "tool": "Bash", "side_effect": se,
                                             "args": {"command": cmd}, "timestamp": ts, **kw}
        steps = [
            {"index": 0, "atif_step_id": 1, "source": "user", "timestamp": "2026-01-01T00:00:00.000Z", "events": []},
            {"index": 1, "atif_step_id": 2, "source": "agent", "timestamp": "2026-01-01T00:00:01.000Z", "events": [
                tc("b1", "python3 train.py", "exec", "2026-01-01T00:00:01.000Z", background_task_id="t1",
                   background_output_file=f)]},
            {"index": 2, "atif_step_id": 3, "source": "agent", "timestamp": "2026-01-01T00:00:05.000Z", "events": [
                {"type": "attachment", "bg_task_id": "t1", "bg_status": "failed", "bg_exit_code": 1,
                 "timestamp": "2026-01-01T00:00:04.000Z"},
                tc("r1", f"cat {f}", "read_only", "2026-01-01T00:00:05.000Z")]},
            {"index": 3, "atif_step_id": 4, "source": "agent", "timestamp": "2026-01-01T00:00:09.000Z", "events": []}]
        plan = replay_plan({"trial": "t", "steps": steps}, 3)
        bg = next(a for a in plan["actions"] if a["call_id"] == "b1")
        self.assertEqual((bg["bg_status"], bg["bg_exit_code"], bg["bg_output_read_later"]), ("failed", 1, ["r1"]))
        rd = next(a for a in plan["actions"] if a["call_id"] == "r1")
        self.assertEqual((rd["side_effect"], rd["evidence_read_for"]), ("exec", "failed_background_output"))

    def test_gate_requires_the_evidence_read_in_the_plan(self):
        a = {"call_id": "b1", "side_effect": "exec_background", "tool": "Bash", "args": {"command": "x"},
             "bg_finished_before_k": True, "bg_status": "failed", "background_output_file": "/tmp/t.output",
             "bg_exit_code": 1, "bg_output_read_later": ["r1"]}
        r = {"call_id": "b1", "fidelity": "bg_rerun_foreground", "rc": 1, "background_output_file": "/tmp/t.output",
             "background_output_sha256": "ab"}
        self.assertFalse(passes({"actions": [a]}, {"results": [r]}), "read not in this plan")
        rd = {"call_id": "r1", "side_effect": "exec", "tool": "Bash", "args": {"command": "cat /tmp/t.output"},
              "orig_text": "Traceback: KeyError", "is_error": False}
        good = {"call_id": "r1", "fidelity": "exact", "rc": 0, "status": "ran", "stdout_full": "Traceback: KeyError\n"}
        self.assertFalse(passes({"actions": [a, rd]}, {"results": [r, good]}))
        other = dict(good, fidelity="differs", stdout_full="Traceback: ModuleNotFoundError\n")
        self.assertFalse(passes({"actions": [a, rd]}, {"results": [r, other]}), "failed for another reason")
        for probe in ("ls -la /tmp/t.output", "test -f /tmp/t.output && echo yes", "stat /tmp/t.output",
                      "grep -l Error /tmp/t.output", "grep -c Error /tmp/t.output", "grep -q Error /tmp/t.output",
                      "grep --count Error /tmp/t.output"):
            ls = dict(rd, args={"command": probe}, orig_text="-rw-r--r-- 1 root root 9 /tmp/t.output")
            res = dict(good, stdout_full="-rw-r--r-- 1 root root 9 /tmp/t.output\n")
            self.assertFalse(passes({"actions": [a, ls]}, {"results": [r, res]}), probe)
        other_file = dict(rd, args={"command": "cat /tmp/other.output"})
        self.assertFalse(passes({"actions": [a, other_file]}, {"results": [r, good]}), "another task's output")


class Pacing(unittest.TestCase):
    def test_unmet_pacing_fails_the_gate(self):
        p, r = one("cat log", "x", "x\n", fidelity="exact")
        r["results"][0]["pacing_unmet"] = True
        self.assertFalse(passes(p, r))

    def test_surviving_job_without_start_time_marks_later_actions_unmet(self):
        with tempfile.TemporaryDirectory() as td:
            plan = {"initial_cwd": td, "actions": [
                {"call_id": "s1", "tool": "Bash", "side_effect": "exec", "timestamp": None,
                 "args": {"command": "nohup sleep 3 >/dev/null 2>&1 &"}, "orig_text": "(Bash completed with no output)"},
                {"call_id": "r1", "tool": "Bash", "side_effect": "exec", "timestamp": "2026-01-01T00:00:01.000Z",
                 "args": {"command": "echo hi"}, "orig_text": "hi"}]}
            pp, rp = Path(td) / "plan.json", Path(td) / "report.json"
            pp.write_text(json.dumps(plan))
            import subprocess
            import sys
            subprocess.run([sys.executable, replay.__file__, str(pp), str(rp), "--exec-cap", "30"], check=False)
            rep = json.loads(rp.read_text())
            self.assertTrue(rep["results"][1].get("pacing_unmet"))
            with self.assertRaises(RuntimeError):
                gate.validate_replay(plan, rep, 0, {})

    def test_pace_wait_reports_unmet_beyond_cap(self):
        import subprocess
        proc = subprocess.Popen(["sleep", "5"], start_new_session=True)
        try:
            job = {"pgid": proc.pid, "replay_t": time.time(), "orig_t": 1000.0}
            a = {"timestamp": "1970-01-01T01:16:40.000Z"}  # orig_t + 3600 s
            w, met = replay.pace_wait([job], a, cap=1)
            self.assertFalse(met)
            w, met = replay.pace_wait([job], {"timestamp": None}, cap=100)
            self.assertFalse(met, "unknown original time while a job is alive")
            w, met = replay.pace_wait([job], {"timestamp": "1970-01-01T00:16:40.300Z"}, cap=10)
            self.assertTrue(met)
        finally:
            proc.kill()


class Replayer(unittest.TestCase):
    def test_empty_output_matches_cc_no_output_text(self):
        self.assertEqual(replay.compare("(Bash completed with no output)", "", 0), "exact")
        self.assertNotEqual(replay.compare("(Bash completed with no output)", "", 1), "exact")

    def test_background_output_trailer_and_pacing(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "task.output"
            now = time.time()
            iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"
            plan = {"initial_cwd": td, "bg_output_trailer": "\n[exited with code {rc}]\n", "actions": [
                {"call_id": "b1", "tool": "Bash", "side_effect": "exec_background",
                 "args": {"command": "echo done"}, "timestamp": iso(now), "bg_finished_before_k": True,
                 "bg_status": "completed", "background_output_file": str(out), "bg_duration_s": 0.1,
                 "orig_text": "Command running in background with ID: b1."},
                {"call_id": "s1", "tool": "Bash", "side_effect": "exec",
                 "args": {"command": f"nohup sh -c 'sleep 2; echo late > {td}/late.txt' >/dev/null 2>&1 &"},
                 "timestamp": iso(now + 1), "orig_text": "(Bash completed with no output)", "duration_s": 0.01},
                {"call_id": "r1", "tool": "Bash", "side_effect": "exec", "args": {"command": f"cat {td}/late.txt"},
                 "timestamp": iso(now + 4), "orig_text": "late", "duration_s": 0.01},
            ]}
            pp, rp = Path(td) / "plan.json", Path(td) / "report.json"
            pp.write_text(json.dumps(plan))
            import subprocess
            import sys
            subprocess.run([sys.executable, replay.__file__, str(pp), str(rp), "--exec-cap", "30"], check=False,
                           env=dict(os.environ), cwd=td)
            rep = json.loads(rp.read_text())
            self.assertEqual(out.read_text(), "done\n\n[exited with code 0]\n")
            res = {r["call_id"]: r for r in rep["results"]}
            self.assertTrue(res["s1"].get("left_background_job"))
            self.assertGreater(res["r1"].get("paced_wait_s", 0), 0)
            self.assertEqual(res["r1"]["fidelity"], "exact")


class Screen(unittest.TestCase):
    def act(self, i, cmd, se="exec", **kw):
        return {"call_id": f"c{i}", "tool": "Bash", "side_effect": se, "args": {"command": cmd}, "atif_step_id": i,
                "timestamp": f"2026-01-01T00:00:{i:02d}.000Z", "orig_text": kw.pop("orig_text", "1.5"), **kw}

    def rules(self, actions, **kw):
        rep = screen.screen_plan({"actions": actions}, kw.pop("rp", None), orig_request={"index": 1}, **kw)
        return rep["verdict"], sorted({(r["severity"], r["rule"]) for r in rep["risks"]})

    def test_clean_prefix_ok(self):
        self.assertEqual(self.rules([self.act(1, "python3 -c 'print(sorted([3,1]))'")])[0], "ok")

    def test_bg_inflight_blocks(self):
        v, r = self.rules([self.act(1, "python3 long.py", se="exec_background", bg_finished_before_k=False)])
        self.assertEqual(v, "block")
        self.assertIn(("block", "bg_inflight"), r)

    def test_read_of_running_bg_output_blocks_finished_is_info(self):
        bg = self.act(1, "python3 long.py", se="exec_background", bg_finished_before_k=True, bg_status="completed",
                      background_output_file="/tmp/t/b1.output", bg_finished_ts="2026-01-01T00:00:05.000Z")
        early = self.act(3, "tail /tmp/t/b1.output")
        late = self.act(9, "cat /tmp/t/b1.output")
        self.assertIn(("block", "pace_dependent_read"), self.rules([bg, early])[1])
        self.assertEqual(self.rules([bg, late])[1], [("info", "pace_dependent_read")])

    def test_generic_rules(self):
        acts = [self.act(1, "python3 -c 'import numpy as np; print(np.random.rand())'"),
                self.act(2, "pip install pypdf"),
                self.act(3, "python3 - <<'EOF'\nimport glob\nprint(glob.glob('/d/*')[0])\nEOF"),
                self.act(4, "python3 - <<'EOF'\nfor x in set(['a','b']): print(x)\nEOF"),
                self.act(5, "nohup python3 train.py > log.txt 2>&1 &"),
                self.act(6, "cat /logs/agent/sessions/projects/-app/x.jsonl")]
        _, r = self.rules(acts, n_compactions=2)
        for want in [("warn", "unseeded_rng"), ("warn", "network"), ("warn", "glob_order"), ("warn", "hash_order"),
                     ("warn", "pace_dependent_read"), ("warn", "agent_state_read"), ("info", "compactions")]:
            self.assertIn(want, r)

    def test_sorted_glob_and_seeded_rng_are_clean(self):
        acts = [self.act(1, "python3 - <<'EOF'\nimport glob\nfor p in sorted(glob.glob('/d/*')): print(p)\nEOF"),
                self.act(2, "python3 -c 'import numpy as np; np.random.seed(0); print(np.random.rand())'")]
        self.assertEqual(self.rules(acts)[0], "ok")


if __name__ == "__main__":
    unittest.main()
