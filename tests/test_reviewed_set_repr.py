"""Reviewed exception: a Python `set` printed directly differs only in repr order (hash seed), never in content."""
import hashlib
import json

from agent_snapshot_resume.replay.gate import validate_replay

CALL = "call_00_test_set_repr"
CMD = "python3 -c \"print(set(['Cz','P3','F6']))\""
OLD = "=== ses-04 bad_amp/bridge {'Cz', 'P3'}\n=== ses-05 {'F6', 'P3', 'Cz'}\n=== ses-06 {'F5', 'P3', 'F6', 'P4'}\n"
NEW = "=== ses-04 bad_amp/bridge {'P3', 'Cz'}\n=== ses-05 {'P3', 'Cz', 'F6'}\n=== ses-06 {'P3', 'P4', 'F5', 'F6'}\n"


def plan():
    return {"actions": [{"call_id": CALL, "tool": "Bash", "side_effect": "exec", "args": {"command": CMD},
                         "orig_text": OLD, "obs_line": 1}]}


def report(stdout):
    return {"results": [{"call_id": CALL, "tool": "Bash", "side_effect": "exec", "status": "ran", "rc": 0,
                         "orig_text": OLD, "orig_full": OLD, "stdout_full": stdout, "stderr_full": "",
                         "fidelity": "differs", "orig_is_error": False}]}


def review(**kw):
    base = {"kind": "python_set_repr_order", "command_sha256": hashlib.sha256(CMD.encode()).hexdigest()}
    return {CALL: {**base, **kw}}


def test_accepts_set_repr_reordering_only():
    assert validate_replay(plan(), report(NEW), "0", review())


def test_rejects_a_real_content_change():
    changed = NEW.replace("'F6'", "'P7'")  # a different member: canonical forms now differ
    try:
        validate_replay(plan(), report(changed), "0", review())
    except RuntimeError as exc:
        assert "Unaccepted replay fidelity" in str(exc)
    else:
        raise AssertionError("a changed set member was accepted")


def test_rejects_without_the_reviewed_command_binding():
    try:
        validate_replay(plan(), report(NEW), "0", review(command_sha256="0" * 64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted without the command hash binding")


def test_rejects_stderr_or_nonzero_rc():
    rep = report(NEW)
    rep["results"][0]["stderr_full"] = "warning\n"
    try:
        validate_replay(plan(), rep, "0", review())
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted with stderr")
    rep = report(NEW)
    rep["results"][0]["rc"] = 1
    try:
        validate_replay(plan(), rep, "0", review())
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted with a non-zero exit code")
