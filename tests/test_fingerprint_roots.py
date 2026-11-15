"""Fingerprint roots: a task that works outside /app must not be fingerprinted at /app."""
from agent_snapshot_resume.bundle import default_fingerprint_roots


def test_app_cwd_keeps_the_default_root():
    assert default_fingerprint_roots({"initial_cwd": "/app", "actions": []}) is None


def test_a_task_workdir_becomes_the_root_when_the_plan_writes_nothing():
    assert default_fingerprint_roots({"initial_cwd": "/task", "actions": []}) == ["/task"]


def test_a_home_cwd_never_binds_the_whole_home():
    assert default_fingerprint_roots({"initial_cwd": "/root", "actions": []}) is None
    assert default_fingerprint_roots({"initial_cwd": "/home/agent", "actions": []}) is None
    # a work directory the run used inside the home is still bound
    assert default_fingerprint_roots({"initial_cwd": "/root/work",
                                      "actions": [{"cwd": "/root/work"}]}) == ["/root/work"]


def test_directories_an_action_used_are_preferred():
    plan = {"initial_cwd": "/task", "actions": [{"cwd": "/task/sub"}, {"args": {"file_path": "/task/sub/x.npy"}}]}
    assert default_fingerprint_roots(plan) == ["/task/sub"]



def test_the_command_runs_on_an_old_task_python():
    """Onsager's task python is 3.9: hashlib.file_digest (3.11+) must not appear in the guest command."""
    from agent_snapshot_resume.replay.fingerprint import build_fingerprint_command
    code = build_fingerprint_command(["/task"])
    assert "file_digest" not in code
    assert "sha256()" in code
