"""A regrade writes a new trial dir; it must never merge into an existing one or into the source trial."""
import pytest

from agent_snapshot_resume.regrade import check_output_dir


def test_a_free_target_is_returned(tmp_path):
    src = tmp_path / "jobs" / "r1b-task-s1-hinteda" / "task__abc"
    src.mkdir(parents=True)
    target = check_output_dir(tmp_path / "regrade", "regrade-task", src)
    assert target == (tmp_path / "regrade" / "regrade-task").resolve()
    assert not target.exists()


def test_an_existing_target_is_refused(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (tmp_path / "regrade" / "regrade-task").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        check_output_dir(tmp_path / "regrade", "regrade-task", src)
    assert "already exists" in str(exc.value)


def test_the_source_trial_itself_is_refused(tmp_path):
    src = tmp_path / "regrade" / "regrade-task"
    src.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        check_output_dir(tmp_path / "regrade", "regrade-task", src)
    assert "must not overlap the source" in str(exc.value)


def test_an_output_dir_above_the_source_is_refused(tmp_path):
    src = tmp_path / "out" / "jobs" / "task__abc"
    src.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        check_output_dir(tmp_path / "out", "jobs", src)
    assert "must not overlap the source" in str(exc.value)
