"""Seeded-file uploads are split so one provider request never carries a whole 40 MB session."""
from agent_snapshot_resume.agents.claude_code import UPLOAD_PART_BYTES, upload_part_ranges


def test_small_files_go_in_one_request():
    assert upload_part_ranges(0) == [(0, 0)]
    assert upload_part_ranges(UPLOAD_PART_BYTES) == [(0, UPLOAD_PART_BYTES)]


def test_large_files_are_split_into_contiguous_parts():
    size = int(4.5 * UPLOAD_PART_BYTES)
    ranges = upload_part_ranges(size, part=1024)
    assert size > 1024
    assert ranges[0][0] == 0 and ranges[-1][1] == size
    assert all(hi - lo <= 1024 and lo < hi for lo, hi in ranges)
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    # every byte is covered exactly once
    body = b"".join(bytes([i % 251]) for i in range(size))
    assert b"".join(body[lo:hi] for lo, hi in ranges) == body
