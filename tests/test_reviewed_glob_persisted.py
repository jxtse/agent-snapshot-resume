"""Reviewed replay exceptions: persisted_output_identical and readonly_unsorted_glob_pick."""
import hashlib
import unittest

from agent_snapshot_resume.replay.gate import validate_replay


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def case(command, orig, fidelity, stdout, review, rc=0, stderr=""):
    plan = {"actions": [{"call_id": "c1", "side_effect": "exec", "tool": "Bash", "args": {"command": command},
                         "orig_text": orig, "is_error": False}]}
    report = {"results": [{"call_id": "c1", "fidelity": fidelity, "stdout_full": stdout, "stderr_full": stderr, "rc": rc}]}
    return plan, report, {"c1": review} if review else {}


FULL = "=== A001\nread_1\t0\tchr3\n" * 200
ORIG = "<persisted-output>\nOutput too large (9KB). Full output saved to: /x/tool-results/b.txt\n\nPreview (first 2KB):\n" + FULL[:2000]
PROBE = "python3 - <<'EOF'\nimport glob\np=glob.glob('/app/data/raster/*/qa.tif')[0]\nprint(p)\nEOF"


class Persisted(unittest.TestCase):
    def test_accept_identical(self):
        cmd = "samtools view a.bam | head"
        p, r, x = case(cmd, ORIG, "persisted_preview", FULL,
                       {"kind": "persisted_output_identical", "command_sha256": sha(cmd), "persisted_output_sha256": sha(FULL)})
        self.assertTrue(validate_replay(p, r, 0, x))

    def test_reject_different_or_unreviewed(self):
        cmd = "samtools view a.bam | head"
        p, r, x = case(cmd, ORIG, "persisted_preview", FULL + "extra",
                       {"kind": "persisted_output_identical", "command_sha256": sha(cmd), "persisted_output_sha256": sha(FULL)})
        with self.assertRaises(RuntimeError):
            validate_replay(p, r, 0, x)
        p, r, x = case(cmd, ORIG, "persisted_preview", FULL, None)
        with self.assertRaises(RuntimeError):
            validate_replay(p, r, 0, x)


class GlobPick(unittest.TestCase):
    def test_accept_readonly(self):
        p, r, x = case(PROBE, "/app/data/raster/b/qa.tif", "differs", "/app/data/raster/a/qa.tif",
                       {"kind": "readonly_unsorted_glob_pick", "command_sha256": sha(PROBE)})
        self.assertTrue(validate_replay(p, r, 0, x))

    def test_reject_writer_or_unbound(self):
        cmd = PROBE.replace("print(p)", "open('/app/out.txt','w').write(p)")
        p, r, x = case(cmd, "", "differs", "", {"kind": "readonly_unsorted_glob_pick", "command_sha256": sha(cmd)})
        with self.assertRaises(RuntimeError):
            validate_replay(p, r, 0, x)
        p, r, x = case(PROBE, "a", "differs", "b", {"kind": "readonly_unsorted_glob_pick", "command_sha256": sha("other")})
        with self.assertRaises(RuntimeError):
            validate_replay(p, r, 0, x)


if __name__ == "__main__":
    unittest.main()
