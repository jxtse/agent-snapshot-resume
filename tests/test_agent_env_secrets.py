"""Credentials in harmless-looking env values must not reach plan.json, job.json or replay."""
import json
import os
from unittest.mock import patch

import pytest

from agent_snapshot_resume import bundle
from agent_snapshot_resume.agent_env import native_agent_env, secret_value
from agent_snapshot_resume.launch import original_agent_settings
from test_review_round7 import pi_bash

SECRETS = {
    "DATABASE_URL": "postgresql://svc:fixture-pass-91@db.internal:5432/app",
    "REDIS_DSN": "redis://:fixture-pass-92@cache:6379/0",
    "PG_CONN": "host=db user=svc password=fixture-pass-93 dbname=app",
    "UPSTREAM": "https://api.example/v1?api_key=fixture-pass-94",
    "HEADER": "Bearer fixturepass95abcdefghijkl",
    "OPAQUE": "Zx9Qm2LpR7vT4kWn8YbC3dHs6JfG1aNe5uXo",
}
PLAIN = {
    "DATA_ROOT": "/data/experiment_2026_11_19/results",
    "HF_ENDPOINT": "https://huggingface.co",
    "DATASET": "s3://bucket/path/to/data",
    "REPO": "git@github.com:org/repo.git",
    "COMMIT": "0123456789abcdef0123456789abcdef01234567",
    "REPORT": "/root/work/RUN_SUMMARY_FINAL_v2_long_name_here.csv",
}
MARKERS = ["fixture-pass-9", "fixturepass95", "Zx9Qm2LpR7vT4kWn8YbC3dHs6JfG1aNe5uXo"]


def write_config(root):
    (root/"config.json").write_text(json.dumps({"agent": {"env": {**SECRETS, **PLAIN}}}))


@pytest.mark.parametrize("value", SECRETS.values())
def test_secret_values_detected(value):
    assert secret_value(value)


@pytest.mark.parametrize("value", PLAIN.values())
def test_plain_values_kept(value):
    assert not secret_value(value)


@pytest.mark.parametrize("agent", ["pi", "codex"])
def test_native_env_drops_secret_values(tmp_path, agent):
    write_config(tmp_path)
    safe, unavailable = native_agent_env(tmp_path, agent)
    assert safe == PLAIN
    assert set(SECRETS) <= unavailable


def test_plan_json_has_no_secret_and_refuses_reference(tmp_path):
    write_config(tmp_path)
    ir = pi_bash(tmp_path, 'psql "$DATABASE_URL" -c "select 1" > out', "")
    plan = bundle.resolved_replay_plan(tmp_path, ir, 2, [])
    written = tmp_path/"plan.json"
    written.write_text(json.dumps(plan))
    text = written.read_text()
    assert not any(m in text for m in MARKERS)
    assert set(plan["shell_env"]) == set(PLAIN)
    assert any("DATABASE_URL" in r for r in plan.get("replay_refusals", []))


def test_claude_code_settings_drop_secret_values(tmp_path):
    write_config(tmp_path)
    env, _ = original_agent_settings(str(tmp_path))
    assert not any(m in json.dumps(env) for m in MARKERS)
    assert env["DATA_ROOT"] == PLAIN["DATA_ROOT"]
    # A template that resolves to a credential on this host is dropped as well.
    (tmp_path/"config.json").write_text(json.dumps({"agent": {"env": {"DB": "${HOST_DB}"}}}))
    with patch.dict(os.environ, {"HOST_DB": SECRETS["DATABASE_URL"]}):
        env, _ = original_agent_settings(str(tmp_path))
    assert "DB" not in env
