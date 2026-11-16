"""The runtime hash collection/gate must record bad files, never abort the whole check."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import pytest

from agent_snapshot_resume.sandbox import runtime_hash as rh
from agent_snapshot_resume.sandbox.runtime_hash import collect_command, hash_problems, parse_collect


def run_collect(paths):
    """Run the real collector script (the guest command, minus the sandbox) and parse it."""
    script = collect_command(paths)
    assert script.startswith("python3 -c ")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return parse_collect(out.stdout)


def test_collect_records_every_bad_path_and_still_hashes_the_good_ones(tmp_path):
    good = tmp_path / "fine.txt"
    good.write_bytes(b"payload")
    (tmp_path / "adir").mkdir()
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    target = tmp_path / "real.txt"
    target.write_text("linked")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    dangling = tmp_path / "gone.txt"
    dangling.symlink_to(tmp_path / "nope.txt")
    missing = tmp_path / "missing.txt"
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    secret.chmod(0o000)

    paths = [str(good), str(tmp_path / "adir"), str(fifo), str(link), str(dangling), str(missing), str(secret)]
    hashes, errors = run_collect(paths)

    assert hashes[str(good)] == hashlib.sha256(b"payload").hexdigest()
    assert hashes[str(link)] == hashlib.sha256(b"linked").hexdigest()  # a symlink hashes its target
    for bad in ("adir", "pipe", "gone.txt", "missing.txt"):
        assert str(tmp_path / bad) in errors, bad
    assert errors[str(tmp_path / "adir")].startswith("NotARegularFile")
    assert errors[str(tmp_path / "missing.txt")].startswith("FileNotFoundError")
    # root can read a 0000 file, an ordinary user cannot: either outcome is recorded, never a crash
    if os.geteuid() != 0:
        assert errors[str(secret)].startswith("PermissionError")
    assert set(hashes) | set(errors) == set(paths)


def test_hash_problems_names_the_file_and_the_reason(tmp_path):
    expected = {"/tests/test.sh": "a" * 64, "/tests/data/x.npy": "b" * 64, "/tests/ok.py": "c" * 64}
    actual = {"/tests/ok.py": "c" * 64, "/tests/test.sh": "d" * 64}
    errors = {"/tests/data/x.npy": "FileNotFoundError: [Errno 2] No such file or directory"}

    problems = hash_problems(expected, actual, errors)
    assert len(problems) == 2
    assert any("/tests/data/x.npy: unreadable (FileNotFoundError" in p for p in problems)
    assert any("/tests/test.sh: sha256 differs" in p for p in problems)
    assert hash_problems(expected, {"/tests/ok.py": "c" * 64}, {}) == ["/tests/data/x.npy: not hashed",
                                                                      "/tests/test.sh: not hashed"]


def test_parse_collect_survives_a_non_dict_payload():
    hashes, errors = parse_collect(json.dumps({"/a": "boom", "/b": {"sha256": "ff", "size": 1}}))
    assert hashes == {"/b": "ff"}
    assert errors == {"/a": "error: boom"}


def test_tests_files_tar_carries_nested_files_and_modes(tmp_path):
    tests = tmp_path / "tests"
    (tests / "verification" / "sub-1").mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (tests / "test.sh").chmod(0o755)
    (tests / "verification" / "sub-1" / "vol.nii.gz").write_bytes(b"\x1f\x8bdata")
    (tests / "data").mkdir()
    (tests / "data" / "population_a.tar.gz").write_bytes(b"packed archive the image unpacks and drops")

    data, n = rh.tests_files_tar_bytes(tests, ["test.sh", "verification/sub-1/vol.nii.gz", "gone.py", "data"])
    assert n == 2  # only the named files that exist: no packed archive, no missing file
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert names == {"test.sh", "verification/sub-1/vol.nii.gz"}
        mode = tar.getmember("test.sh").mode
    assert mode & 0o111  # restored test.sh stays executable


class FakeSandbox:
    def __init__(self):
        self.written = {}
        self.files = self
        self.sandbox_id = "sbx-test"

    async def write(self, path, data):
        self.written[path] = data


def _env(tmp_path, environment_dir: Path, collector_stdout: str):
    from agent_snapshot_resume.sandbox.snapshot_env import SnapshotCloneEnvironment

    env = SnapshotCloneEnvironment.__new__(SnapshotCloneEnvironment)
    env.environment_dir = environment_dir
    env._sandbox = FakeSandbox()
    env.session_id = "trial__verifier__k"
    calls = {"collect": 0, "commands": []}

    class Paths:
        trial_dir = tmp_path

    env.trial_paths = Paths()

    async def fake_collect(_expected):
        calls["collect"] += 1
        return parse_collect(collector_stdout)

    async def fake_exec(command, timeout_sec=None, user=None):
        calls["commands"].append(command)
        class R:
            return_code = 0
            stdout = ""
            stderr = ""
        return R()

    env._collect_runtime_hashes = fake_collect
    env.exec = fake_exec
    return env, calls


def test_verifier_check_repairs_an_incomplete_tests_dir(tmp_path):
    tests = tmp_path / "task-tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\n")
    (tests / "test.sh").chmod(0o755)
    expected = {"/tests/test.sh": hashlib.sha256(b"#!/bin/sh\n").hexdigest()}
    incomplete = json.dumps({"/tests/test.sh": {"error": "FileNotFoundError", "detail": "No such file"}})
    env, calls = _env(tmp_path, tests, incomplete)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(env._check_runtime_hashes(expected, "verifier"))

    # repaired once (upload + extract) and then re-checked: the error names the file, not a guest traceback
    assert calls["collect"] == 2
    assert any("tar -xzf" in c for c in calls["commands"])
    assert "/tests/test.sh: unreadable (FileNotFoundError" in str(exc.value)
    record = json.loads((tmp_path / "infra-validation" / "trial__verifier__k-verifier.json").read_text())
    assert record["repaired"]["files"] == 1
    assert "FileNotFoundError" in record["errors"]["/tests/test.sh"]


def test_verifier_check_passes_after_a_successful_repair(tmp_path):
    tests = tmp_path / "task-tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\n")
    expected = {"/tests/test.sh": hashlib.sha256(b"#!/bin/sh\n").hexdigest()}
    outputs = [json.dumps({"/tests/test.sh": {"error": "NotARegularFile", "detail": "0o40000"}}),
               json.dumps({"/tests/test.sh": {"sha256": expected["/tests/test.sh"], "size": 10}})]
    env, calls = _env(tmp_path, tests, outputs[0])
    env._collect_runtime_hashes = _sequence(outputs)

    assert asyncio.run(env._check_runtime_hashes(expected, "verifier")) is True
    record = json.loads((tmp_path / "infra-validation" / "trial__verifier__k-verifier.json").read_text())
    assert record["actual"] == expected and record["errors"] == {}


def _sequence(outputs):
    remaining = list(outputs)

    async def fake_collect(_expected):
        return parse_collect(remaining.pop(0))
    return fake_collect


def test_auxiliary_check_never_repairs(tmp_path):
    expected = {"/app/out/bg.txt": "e" * 64}
    env, calls = _env(tmp_path, tmp_path, json.dumps({"/app/out/bg.txt": {"error": "FileNotFoundError",
                                                                        "detail": "gone"}}))
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(env._check_runtime_hashes(expected, "auxiliary"))
    assert calls["collect"] == 1
    assert calls["commands"] == []
    assert "auxiliary runtime files do not match" in str(exc.value)
