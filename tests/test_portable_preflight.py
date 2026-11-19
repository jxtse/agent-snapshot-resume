import json
import shutil
import sqlite3

import pytest

from agent_snapshot_resume import cli, config
from agent_snapshot_resume.preflight import CAPABILITIES, CONTEXT, compare_exports
from agent_snapshot_resume.trace.claude_code import align_user_notifications


def export(root, database, service):
    directory = root / "services" / service
    directory.mkdir(parents=True)
    for key in CAPABILITIES:
        (directory / key).write_text(f"{service}: {key}")
    (directory / "checkpoint").write_bytes(database.read_bytes())
    with sqlite3.connect(database) as connection:
        observed, = connection.execute("select count(*) from items").fetchone()
    (directory / "output").write_text(str(observed))


@pytest.fixture
def captures(tmp_path):
    roots = [tmp_path / "original", tmp_path / "restore"]
    for service, count in (("catalog", 2), ("queue", 3)):
        source = tmp_path / f"{service}.db"
        with sqlite3.connect(source) as db:
            db.execute("create table items (id integer)")
            db.executemany("insert into items values (?)", [(n,) for n in range(count)])
        export(roots[0], source, service)
        clone = tmp_path / f"restored-{service}.db"
        shutil.copyfile(roots[0] / "services" / service / "checkpoint", clone)
        export(roots[1], clone, service)
    for root in roots:
        (root / "topology.json").write_text(json.dumps({"services": ["catalog", "queue"]}))
        for key in (*CONTEXT, "harness"):
            (root / key).write_text(key)
        (root / "request.json").write_text(json.dumps({
            "model": "fixture", "system": "system", "tools": [{"name": "list"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "items"}]}],
        }))
    return roots


def test_cli_two_services(captures, monkeypatch, capsys):
    monkeypatch.setattr(config, "load_env_file", lambda *args: pytest.fail("loaded credentials"))
    assert cli.main(["preflight", *map(str, captures)]) == 0
    assert json.loads(capsys.readouterr().out)["historical_restore_verified"] is False


@pytest.mark.parametrize("key", ["services/queue", "services/queue/processes", "children", "tools", "prefix",
                                 "system", "effort", "deadline", "cut"])
def test_missing_evidence(captures, key):
    path = captures[1] / key
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    assert compare_exports(*captures)["comparison"] == "blocked"


@pytest.mark.parametrize("key", ["binaries", "output", "checkpoint", "limits", "volumes"])
def test_changed_observation(captures, key):
    (captures[1] / "services/catalog" / key).write_bytes(b"mutated")
    assert compare_exports(*captures)["environment_problems"] == [f"catalog/{key}"]


def test_ambiguous_alias(captures):
    # Both candidates have identical bytes but neither preserves the recorded namespace.
    files = "run-21.log  abc\n"
    (captures[0] / "services/catalog/files").write_text(files)
    (captures[1] / "services/catalog/files").write_text("run-81.log  abc\nrun-82.log  abc\n")
    assert compare_exports(*captures)["comparison"] == "blocked"


def test_positioned_hint_and_complete_request(captures):
    ref, restored = captures
    hint = {"message": 0, "block": 1, "text": "declared"}
    request = restored / "request.json"
    assert compare_exports(ref, restored, hint)["context_problems"] == ["first_request"]
    body = json.loads(request.read_text())
    body["messages"][0]["content"].append({"type": "text", "text": hint["text"]})
    request.write_text(json.dumps(body))
    assert compare_exports(ref, restored, hint)["comparison"] == "captures-match"
    assert compare_exports(ref, restored, dict(hint, block=0))["comparison"] == "blocked"
    body["tools"][0]["name"] = "changed"
    request.write_text(json.dumps(body))
    assert compare_exports(ref, restored, hint)["comparison"] == "blocked"


def test_changed_harness_control(captures, tmp_path):
    (captures[1] / "harness").write_text("adapter")
    assert compare_exports(*captures)["comparison"] == "blocked"
    control = tmp_path / "control"
    shutil.copytree(captures[1], control)
    result = compare_exports(*captures, control=control)
    assert result["comparison"] == "captures-match"
    assert result["condition"] == "alternate-harness-diagnostic"


def test_shared_evidence_refused(captures):
    assert compare_exports(captures[0], captures[0])["comparison"] == "blocked"


def test_native_alignment_requires_unique_exact_adjacent_user_pair():
    records = [{"message": {"content": t}} for t in ("one", "two")]
    native = [dict(source="user", timestamp=str(i), line_start=i) for i in range(2)]
    exported = [dict(source="user", step_id=i+1, timestamp=str(i), message=r["message"]["content"])
                for i, r in enumerate(records)]
    assert align_user_notifications(native, exported[::-1], records) == exported
    for field, value in (("source", "agent"), ("timestamp", "0"), ("message", "changed"), ("step_id", 1)):
        assert align_user_notifications(native, [dict(exported[1], **{field: value}), exported[0]], records) is None

def test_same_service_omitted_from_both_exports(captures):
    for root in captures:
        shutil.rmtree(root / "services/queue")
    assert compare_exports(*captures)["environment_problems"] == ["topology"]


def test_duplicate_request_key(captures):
    path = captures[1] / "request.json"
    path.write_text(path.read_text().replace('"model":', '"model":"other","model":'))
    assert compare_exports(*captures)["context_problems"] == ["first_request"]
