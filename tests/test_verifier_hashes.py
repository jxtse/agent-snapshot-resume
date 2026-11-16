"""The verifier's /tests identity: build context, caches and packed inputs are not verifier runtime files."""
import hashlib
from pathlib import Path

from agent_snapshot_resume.launch import PACKED_INPUT_SUFFIXES, verifier_hashes


def test_packed_inputs_are_not_part_of_the_tests_identity(tmp_path):
    tests = tmp_path / "tests"
    (tests / "data").mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/sh\n")
    (tests / "test_state.py").write_text("print('ok')\n")
    (tests / "data" / "suite_manifest.json").write_text("{}")
    (tests / "data" / "population_a.tar.gz").write_bytes(b"packed source the image unpacks and drops")
    (tests / "Dockerfile").write_text("FROM x\n")
    (tests / ".dockerignore").write_text("*.pyc\n")
    (tests / "__pycache__").mkdir()
    (tests / "__pycache__" / "x.pyc").write_bytes(b"junk")

    hashes = verifier_hashes(tmp_path)
    assert set(hashes) == {"/tests/test.sh", "/tests/test_state.py", "/tests/data/suite_manifest.json"}
    assert hashes["/tests/test.sh"] == hashlib.sha256(b"#!/bin/sh\n").hexdigest()


def test_a_packed_suffix_is_recognised_case_insensitively():
    assert Path("data.ZIP").name.lower().endswith(PACKED_INPUT_SUFFIXES)
    assert Path("data.tar.gz").name.lower().endswith(PACKED_INPUT_SUFFIXES)
    assert not Path("test_outputs.py").name.lower().endswith(PACKED_INPUT_SUFFIXES)
