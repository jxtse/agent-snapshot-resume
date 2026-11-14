"""reviewed_masked_output: the reviewed escape hatch for outputs that only differ in volatile/machine tokens.

Bound to the command's sha256, accepted only when
  * the masked outputs carry the same information (same lines modulo dates, addresses, UUIDs, machine-spec rows,
    `ls` block counts and `./..` rows, and numbers),
  * the exit outcome is unchanged and the action actually ran (a timeout is never masked away),
  * the reviewer recorded that no changed number reappears in the agent's later reasoning/tool arguments, and
  * filesystem order may be ignored only for listing commands (`ls/find/du/tree/stat/wc`).
"""
import hashlib
import unittest

from agent_snapshot_resume.replay.gate import (MASK_VERSION, mask_volatile, masked_equal, validate_replay)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


CMD = "nproc; free -g; ls -la /app/work | head -20"
ORIG = """4
               total        used        free
Mem:              15           1          13
total 1040
-rw-rw-r-- 1 root root 2272 Nov  3 14:35 back_3B.png
-rw-r--r-- 1 root root 2136 Nov  3 14:35 nadir_3N.png
"""
REPLAY = """2
               total        used        free
Mem:               3           0           1
total 1048
drwxr-xr-x  2 root root 4096 Nov 13 10:16 .
-rw-rw-r--  1 root root 2272 Nov 13 10:16 back_3B.png
-rw-r--r--  1 root root 2136 Nov 13 10:16 nadir_3N.png
"""


def run(fidelity="differs", orig=ORIG, replay=REPLAY, rc=0, review=None, command=CMD, status="ran",
        orig_is_error=False):
    plan = {"actions": [{"call_id": "call_00_x", "side_effect": "exec", "orig_text": orig,
                         "args": {"command": command}, "is_error": orig_is_error}]}
    report = {"results": [{"call_id": "call_00_x", "fidelity": fidelity, "status": status, "rc": rc,
                           "stdout_full": replay, "stderr_full": ""}]}
    return validate_replay(plan, report, "0", {"call_00_x": review} if review else {})


def masked_review(order="ordered", reused_later=False, command=CMD):
    return {"kind": "reviewed_masked_output", "mask_version": MASK_VERSION, "order": order,
            "command_sha256": sha(command), "changed_numbers_reused_later": reused_later, "note": "reviewed"}


class MaskShape(unittest.TestCase):
    def test_machine_and_volatile_tokens_are_normalised(self):
        m = mask_volatile(ORIG + "0x7f8a1b2c3d4e 2026-11-13T10:16:00Z 9f6a2b1c-1111-2222-3333-444455556666\n"
                                  "/opt/tbs-proxy/src/passthrough_proxy.py\n")
        self.assertEqual(m, ["<n>", "total used free", "<mem>",
                             "-rw-rw-r-- <n> root root <n> <date> back_3B.png",
                             "-rw-r--r-- <n> root root <n> <date> nadir_3N.png",
                             "<addr> <date> <uuid>", "/opt/<proxy-dir>/src/passthrough_proxy.py"])

    def test_identifiers_with_digits_survive(self):
        self.assertEqual(mask_volatile("netCDF4._netCDF4 python3.11"), ["netCDF4._netCDF4 python3.11"])

    def test_structural_text_is_not_masked(self):
        self.assertFalse(masked_equal("ModuleNotFoundError: No module named 'geopandas'",
                                      "ModuleNotFoundError: No module named 'numpy'"))
        self.assertFalse(masked_equal("/opt/conda/envs/stereo/bin/python3", "/opt/conda/bin/python3"))

    def test_order_modes(self):
        self.assertFalse(masked_equal("a\nb", "b\na"))
        self.assertTrue(masked_equal("a\nb", "b\na", "sorted"))
        self.assertTrue(masked_equal("x\na\nb", "x\na\nb\nc", "subsequence"))
        self.assertFalse(masked_equal("a\nb\nc", "c\na\nb", "subsequence"))


class ReviewedMaskedOutput(unittest.TestCase):
    def test_accepts_masked_machine_noise(self):
        run(review=masked_review())

    def test_rejects_when_numbers_are_reused_later(self):
        with self.assertRaises(RuntimeError) as ctx:
            run(review=masked_review(reused_later=True))
        self.assertIn("call_00_x", str(ctx.exception))

    def test_rejects_unbound_command(self):
        review = dict(masked_review(), command_sha256="0" * 64)
        with self.assertRaises(RuntimeError):
            run(review=review)

    def test_rejects_unknown_mask_version(self):
        review = dict(masked_review(), mask_version=MASK_VERSION + 1)
        with self.assertRaises(RuntimeError):
            run(review=review)

    def test_rejects_real_text_difference(self):
        replay = REPLAY.replace("nadir_3N.png", "nadir_3N.tif")
        with self.assertRaises(RuntimeError):
            run(replay=replay, review=masked_review())

    def test_never_masks_a_timeout(self):
        with self.assertRaises(RuntimeError):
            run(fidelity="timeout", status="timeout", review=masked_review())

    def test_exit_outcome_must_match(self):
        with self.assertRaises(RuntimeError):
            run(rc=1, orig_is_error=False, review=masked_review())

    def test_sorted_order_only_for_listings(self):
        orig, replay = "a.png\nb.png\n", "b.png\na.png\n"
        with self.assertRaises(RuntimeError):
            run(orig=orig, replay=replay, command="python run.py", review=masked_review(order="sorted"))
        cmd = "find /app -name '*.png'"
        run(orig=orig, replay=replay, command=cmd, review=masked_review(order="sorted", command=cmd))

    def test_without_exception_it_still_fails(self):
        with self.assertRaises(RuntimeError):
            run()


if __name__ == "__main__":
    unittest.main()
