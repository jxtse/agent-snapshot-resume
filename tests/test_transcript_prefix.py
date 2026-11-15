"""The replay environment carries the parent's own transcript at the cut, not the normalized seed."""
import json
from pathlib import Path

from agent_snapshot_resume.sandbox.source import transcript_prefix


def _trial(root: Path, raw: list[str]) -> Path:
    trial = root / "jobs" / "r1-x"
    proj = trial / "agent" / "sessions" / "projects" / "-app"
    proj.mkdir(parents=True)
    (proj / "sid.jsonl").write_text("\n".join(raw) + "\n")
    return trial


def test_the_parent_prefix_includes_the_scaffolding_the_seed_strips(tmp_path):
    raw = [json.dumps({"type": "user", "n": i}) for i in range(20)]
    parent = _trial(tmp_path, raw)
    bdir = tmp_path / "bundle"
    bdir.mkdir()
    # the seed is the parent's lines minus the six scaffolding records at 6..11 (line numbers of the raw file)
    seed = [raw[i] for i in range(20) if not (6 <= i < 12)]
    (bdir / "session_trunc.jsonl").write_text("\n".join(seed[:14]) + "\n")
    meta = {"cut_line": 14,  # 20 - 6 removed lines before it
            "chain": {"parent_trial_dir": str(parent),
                      "stripped": [{"removed_lines": [6, 7, 8, 9, 10, 11], "reason": "resume scaffolding"}]}}
    got = transcript_prefix(meta, bdir).decode().splitlines()
    assert got == raw[:20]
    assert len(got) == 20


def test_without_a_parent_session_the_seed_is_used(tmp_path):
    bdir = tmp_path / "bundle"
    bdir.mkdir()
    (bdir / "session_trunc.jsonl").write_text("{}\n")
    assert transcript_prefix({"cut_line": 1, "chain": {"parent_trial_dir": str(tmp_path / "gone")}},
                             bdir).decode() == "{}\n"


def test_a_cut_right_next_to_a_removed_range_maps_back(tmp_path):
    """removed [6, 12) with the cut at original line 12 (normalized 6): the answer must be 12, not 6."""
    raw = [json.dumps({"type": "user", "n": i}) for i in range(20)]
    parent = _trial(tmp_path, raw)
    bdir = tmp_path / "bundle"
    bdir.mkdir()
    (bdir / "session_trunc.jsonl").write_text("{}\n")
    meta = {"cut_line": 6, "chain": {"parent_trial_dir": str(parent),
                                     "stripped": [{"removed_lines": [6, 7, 8, 9, 10, 11]}]}}
    assert transcript_prefix(meta, bdir).decode().splitlines() == raw[:12]
