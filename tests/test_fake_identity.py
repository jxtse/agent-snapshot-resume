"""A fake run's recorded bundle identity must accept a record written before derived content keys, and nothing else."""
from agent_snapshot_resume.bundle import bundle_identity_diff, orig_request_diff

ORIG = {"index": 66, "method": "usage", "req_bytes": 451509, "model": "deepseek-flash", "n_messages": 80,
        "max_tokens": 128000, "thinking": {"type": "adaptive"}, "n_tools": 12,
        "content": {"message_block_sha256": ["a" * 64, "b" * 64],
                    "block_kinds": [["user", "text"], ["assistant", "tool_use"]],
                    "meta_block_sha256": [],
                    "n_attachments": 0, "attachments": [], "rendered_templates": None}}


def older(orig):
    """The same reference as a fake run recorded it before block_kinds/meta_block_sha256 existed."""
    content = {k: v for k, v in orig["content"].items() if k not in ("block_kinds", "meta_block_sha256")}
    return {**orig, "content": content}


def test_a_record_without_the_derived_content_keys_is_identical():
    assert orig_request_diff(ORIG, older(ORIG)) == []
    assert orig_request_diff(older(ORIG), ORIG) == []  # and the reverse: nothing the other side lacks


def test_any_other_content_difference_is_refused():
    changed = older(ORIG)
    changed["content"] = {**changed["content"], "message_block_sha256": ["a" * 64, "c" * 64]}
    assert orig_request_diff(ORIG, changed) == ["content.message_block_sha256"]

    dropped = older(ORIG)
    dropped["content"] = {k: v for k, v in dropped["content"].items() if k != "n_attachments"}
    assert orig_request_diff(ORIG, dropped) == ["content.n_attachments"]  # mine has a key theirs does not


def test_a_different_original_request_is_refused():
    changed = older(ORIG)
    changed["req_bytes"] = 451510
    assert orig_request_diff(ORIG, changed) == ["req_bytes"]
    assert orig_request_diff(None, ORIG) == ["orig_request_at_k"]
    assert orig_request_diff(ORIG, None) == ["orig_request_at_k"]


CUT = {"trial": "t", "session_sha256": "s" * 64, "cut_line": 392, "resume_atif_step": 68, "agent": "claude-code",
       "base_snapshot": None, "agent_version_orig": "2.1.280", "fingerprint_roots": None,
       "fingerprint_optional_roots": [], "plan_sha256": "p" * 64, "replay_reviewed_exceptions": {}}

IDENT = {"cut": CUT, "seed_sha256": "e" * 64, "proxy_spec_sha256": "x" * 64, "orig_request_at_k": ORIG,
         "run_id": "task-s392-hint"}


def test_identity_accepts_both_compatibility_rules():
    theirs = {"cut": {k: v for k, v in CUT.items() if k != "fingerprint_optional_roots"},
              "seed_sha256": "e" * 64, "proxy_spec_sha256": "x" * 64,
              "orig_request_at_k": older(ORIG), "run_id": "task-s392-hint"}
    assert bundle_identity_diff(IDENT, theirs) == []


def test_identity_still_refuses_a_different_seed_or_request():
    assert bundle_identity_diff(IDENT, {**IDENT, "seed_sha256": "f" * 64}) == ["seed_sha256"]
    other = {**IDENT, "orig_request_at_k": {**older(ORIG), "n_tools": 13}}
    assert bundle_identity_diff(IDENT, other) == ["orig_request_at_k"]
    assert bundle_identity_diff(IDENT, {}) == ["cut", "orig_request_at_k", "proxy_spec_sha256", "run_id",
                                               "seed_sha256"]
