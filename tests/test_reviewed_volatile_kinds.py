"""Reviewed exception kinds for ls ./.. entries and unseeded RNG probes (command-bound, fail-closed)."""
import hashlib
import unittest

from agent_snapshot_resume.replay.gate import validate_replay

OLD_LS = "total 64132\n-rw-rw-r-- 1 root root 65587413 Oct 13 04:38 a.h5\n---\ntotal 4\ndrwxr-xr-x 2 root root 4096 Oct 29 18:17 data"
NEW_LS = ("total 64140\ndrwxr-xr-x 2 root root     4096 Oct 29 18:17 .\ndrwxr-xr-x 3 root root     4096 Oct 29 18:18 ..\n"
          "-rw-rw-r-- 1 root root 65587413 Oct 13 04:38 a.h5\n---\ntotal 12\ndrwxr-xr-x  3 root root 4096 Oct 29 18:18 .\n"
          "drwxr-xr-x 21 root root 4096 Nov 10 08:51 ..\ndrwxr-xr-x  2 root root 4096 Oct 29 18:17 data")


def run(kind, command, old, new, fidelity="differs", rc=0, err="", bind=True):
    a = {"call_id": "x", "tool": "Bash", "side_effect": "exec", "is_error": False, "args": {"command": command},
         "orig_text": old}
    r = {"call_id": "x", "fidelity": fidelity, "rc": rc, "stdout_full": new, "stderr_full": err}
    sha = hashlib.sha256((command if bind else "other").encode()).hexdigest()
    return validate_replay({"actions": [a]}, {"results": [r]}, 0, {"x": {"kind": kind, "command_sha256": sha}})


class VolatileKinds(unittest.TestCase):
    def test_ls_dot_entries(self):
        cmd = "ls -la /app/data/ && echo --- && ls -la /app/"
        self.assertTrue(run("ls_dot_entries", cmd, OLD_LS, NEW_LS))
        with self.assertRaises(RuntimeError):  # a real file row changed
            run("ls_dot_entries", cmd, OLD_LS, NEW_LS.replace("65587413", "65587414"))
        with self.assertRaises(RuntimeError):  # extra file
            run("ls_dot_entries", cmd, OLD_LS, NEW_LS + "\n-rw-r--r-- 1 root root 1 Nov 10 08:51 new.txt")
        with self.assertRaises(RuntimeError):  # not bound to this command
            run("ls_dot_entries", cmd, OLD_LS, NEW_LS, bind=False)
        with self.assertRaises(RuntimeError):  # not an ls command
            run("ls_dot_entries", "cat x", OLD_LS, NEW_LS)

    def test_unseeded_rng(self):
        cmd = "python3 -c 'import numpy as np; print(np.random.choice([-1,1],10).mean())'"
        self.assertTrue(run("unseeded_rng_numbers", cmd, "negfrac 0.5015 tot 10648", "negfrac 0.4971 tot 10648",
                            fidelity="numbers_differ"))
        with self.assertRaises(RuntimeError):
            run("unseeded_rng_numbers", cmd, "negfrac 0.5 tot 1", "posfrac 0.5 tot 1", fidelity="numbers_differ")
        with self.assertRaises(RuntimeError):
            run("unseeded_rng_numbers", cmd.replace("np.random", "np.random.seed(0); np.random"), "a 1", "a 2",
                fidelity="numbers_differ")
