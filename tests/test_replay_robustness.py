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


def one(command, orig, stdout, stderr="", rc=0, is_error=False, fidelity="differs", side_effect="exec", **extra):
    a = {"call_id": "c1", "side_effect": side_effect, "tool": "Bash", "args": {"command": command},
         "orig_text": orig, "is_error": is_error, **extra}
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


class AutomaticEquivalence(unittest.TestCase):
    def test_ls_dot_entries(self):
        self.assertTrue(passes(*one("ls -la /app/data", LS_OLD, LS_NEW)))
        changed = LS_NEW.replace("7 Oct", "8 Oct")
        self.assertFalse(passes(*one("ls -la /app/data", LS_OLD, changed)))
        self.assertFalse(passes(*one("cat x", LS_OLD, LS_NEW)), "only for ls -l listings")

    def test_timing_lines(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('loss 0.25')\nprint('time', time.time()-t)\nEOF"
        self.assertTrue(passes(*one(cmd, "loss 0.25\ntime 0.851", "loss 0.25\ntime 0.862")))
        self.assertFalse(passes(*one(cmd, "loss 0.25\ntime 0.851", "loss 0.26\ntime 0.862")), "science number")
        self.assertFalse(passes(*one(cmd, "per-time err 0.1", "per-time err 0.2")), "label is not a duration")
        self.assertFalse(passes(*one("python3 -c 'print(1)'", "time 0.851", "time 0.862")), "command does not time")

    def test_exit_outcome_must_match(self):
        cmd = "python3 - <<'EOF'\nimport time\nt=time.time()\nprint('time', time.time()-t)\nEOF"
        p, r = one(cmd, "time 0.851", "time 0.862", rc=1)
        self.assertFalse(passes(p, r))

    def test_grep_no_match_rc1(self):
        cmd = "python3 run.py\npip list | grep -i pypdf"
        self.assertTrue(passes(*one(cmd, "saved 22", "saved 22\n", rc=1)))
        self.assertFalse(passes(*one(cmd, "saved 22", "saved 23\n", rc=1)))
        self.assertFalse(passes(*one("python3 run.py", "saved 22", "saved 22\n", rc=1)), "not a grep pipeline")

    def test_readonly_listing_order(self):
        cmd = "python3 - <<'EOF'\nimport glob\nfor d in glob.glob('/d/*'):\n    print(d, end=' ')\nEOF"
        self.assertTrue(passes(*one(cmd, "/d/a /d/b /d/c", "/d/b /d/a /d/c")))
        writes = cmd.replace("print(d, end=' ')", "open('/tmp/o','w').write(d)")
        self.assertFalse(passes(*one(writes, "/d/a /d/b", "/d/b /d/a")), "writes depend on the order")
        self.assertFalse(passes(*one(cmd, "/d/a /d/b", "/d/b /d/x")), "different items")

    def test_shell_diagnostic_prefix(self):
        o = "Exit code 2\nout 1\n/bin/bash: eval: line 18: unexpected EOF while looking for matching `\"'"
        n = "out 1\nbash: -c: line 9: unexpected EOF while looking for matching `\"'"
        self.assertTrue(passes(*one('python3 -c "x', o, "out 1\n", stderr=n.split("\n")[1], rc=2, is_error=True)))
        self.assertFalse(passes(*one('python3 -c "x', o, "out 2\n", stderr=n.split("\n")[1], rc=2, is_error=True)))

    def test_shell_time_report(self):
        cmd = "TIMEFORMAT='%3R s'; time python3 solver.py; ls out"
        self.assertTrue(passes(*one(cmd, "6.098 s\nfit ok\na.json", "fit ok\na.json\n", stderr="6.103 s\n")))
        self.assertFalse(passes(*one(cmd, "6.098 s\nfit ok\na.json", "fit bad\na.json\n", stderr="6.103 s\n")))
        cmd2 = "time python3 solver.py"
        self.assertTrue(passes(*one(cmd2, "real\t0m6.7s\nuser\t0m6.8s\nsys\t0m0.4s\nok", "ok\n",
                                    stderr="\nreal\t0m6.9s\nuser\t0m6.9s\nsys\t0m0.5s\n")))

    def test_cprofile_timings(self):
        cmd = "python3 -m cProfile -s cumtime solver.py | head -8"
        o = ("  981 function calls (979 primitive calls) in 38.063 seconds\n\n   Ordered by: cumulative time\n\n"
             "   ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n"
             "    704/1    0.006    0.000   38.102   38.102 {built-in method builtins.exec}")
        n = o.replace("38.063", "38.886").replace("38.102   38.102", "38.925   38.925")
        self.assertTrue(passes(*one(cmd, o, n)))
        self.assertFalse(passes(*one(cmd, o, n.replace("704/1", "705/1"))), "call counts differ")

    def test_agent_state_read(self):
        cmd = "python3 - <<'EOF'\nimport json\nfor l in open('/logs/agent/sessions/projects/-app/x.jsonl'): print(l)\nEOF"
        err = "FileNotFoundError: [Errno 2] No such file or directory: '/logs/agent/sessions/projects/-app/x.jsonl'\n"
        self.assertTrue(passes(*one(cmd, "Reconstruct the ...", "", stderr=err, rc=1)))
        w = cmd.replace("print(l)", "open('/app/o','w').write(l)")
        self.assertFalse(passes(*one(w, "Reconstruct the ...", "", stderr=err, rc=1)), "writes")

    def test_volatile_and_order(self):
        o = "<stdin>:15: RuntimeWarning: x\ndrwxr-xr-x 2 root root 4096 Nov  2 20:17 .\nres 1"
        n = "drwxr-xr-x 2 root root 4096 Nov 13 04:47 .\nres 1\n"
        self.assertTrue(passes(*one("python3 a.py; ls -la d", o, n, stderr="<stdin>:15: RuntimeWarning: x\n")))


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
    def bg(self, status, rc):
        a = {"call_id": "c1", "side_effect": "exec_background", "tool": "Bash", "args": {"command": "x"},
             "bg_finished_before_k": True, "bg_status": status, "background_output_file": "/tmp/t.output"}
        r = {"call_id": "c1", "fidelity": "bg_rerun_foreground", "rc": rc, "background_output_file": "/tmp/t.output",
             "background_output_sha256": "ab"}
        return {"actions": [a]}, {"results": [r]}

    def test_failed_task_reproduced_failed(self):
        self.assertTrue(passes(*self.bg("failed", 1)))
        self.assertFalse(passes(*self.bg("failed", 0)))
        self.assertTrue(passes(*self.bg("completed", 0)))
        self.assertFalse(passes(*self.bg("completed", 1)))


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
