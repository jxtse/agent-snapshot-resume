"""Write a Harbor job config for one snapshot+session resume trial, and optionally run it.

The job config never contains a credential: the model key is read by the agent from ASR_UPSTREAM_KEY in the Harbor
process environment at run time and passed into the sandbox over a pipe (see proxy/runtime.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import config
from .agents.base import load_bundle

SNAPSHOT_ENV_IMPORT = "agent_snapshot_resume.sandbox.snapshot_env:SnapshotCloneEnvironment"
_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_RE = re.compile(r"(KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH)", re.I)
# Agent env keys that must come from this tool, never from the original run.
_RESERVED_ENV = {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_BASE_URL",
                 "OPENAI_API_KEY", "CLAUDE_CONFIG_DIR"}


def verifier_hashes(task_dir: Path) -> dict[str, str]:
    tests = task_dir / "tests"
    return {"/tests/" + str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(tests.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.name not in ("Dockerfile", ".dockerignore")}


def original_agent_settings(trial_dir: str | None) -> tuple[dict[str, str], dict[str, Any]]:
    """Non-secret env and kwargs of the original run's agent (from <trial>/config.json), to reproduce its setup."""
    if not trial_dir or not (Path(trial_dir) / "config.json").is_file():
        return {}, {}
    agent = (json.loads((Path(trial_dir) / "config.json").read_text()).get("agent") or {})
    env = {}
    for k, v in (agent.get("env") or {}).items():
        if k in _RESERVED_ENV or _SECRET_RE.search(k):
            continue
        m = _TEMPLATE_RE.fullmatch(str(v))
        if m:  # "${VAR}" / "${VAR:-default}": resolve from this process (non-secret names only), else drop
            v = os.environ.get(m.group(1)) or m.group(2)
            if not v:
                print(f"warning: {k}={agent['env'][k]} not set here; dropped", file=sys.stderr)
                continue
        env[k] = str(v)
    kwargs = {k: v for k, v in (agent.get("kwargs") or {}).items() if k != "version" and not _SECRET_RE.search(k)}
    return env, kwargs


def check_shared_snapshot(source: dict[str, Any], bundle: Path) -> None:
    """Allow a snapshot prepared from another bundle only if both bundles cut the same trial at the same point with
    the same replay plan (snapshot_cut_identity); otherwise refuse."""
    from .bundle import snapshot_cut_identity

    mine = snapshot_cut_identity(bundle)
    theirs = source.get("cut_identity")
    if theirs is None and source.get("bundle") and (Path(source["bundle"]) / "bundle.json").is_file():
        theirs = snapshot_cut_identity(Path(source["bundle"]))
    if theirs != mine:
        raise SystemExit("snapshot state was prepared from a different bundle (cut identity differs)")
    if source.get("agent_version") and mine.get("agent_version_orig") and \
            source["agent_version"] != mine["agent_version_orig"]:
        raise SystemExit(f"snapshot has agent {source['agent_version']}, bundle needs {mine['agent_version_orig']}")
    print(f"note: sharing snapshot {source.get('snapshot_id')} prepared from bundle {source['run_id']} "
          "(identical cut identity)", file=sys.stderr)


FREEZE_IDENTITY_KEYS = ("run_id", "session_sha256", "cut_line", "sentinel", "mode", "hint", "agent_version_orig",
                        "orig_request_at_k")


def frozen_first_request_digest(path: Path, meta: dict[str, Any], passthrough_args: list[str] | None) -> str:
    """Digest of a fake run's rewritten first request (every field except `metadata`), to require the real run's
    first request to be identical. Verified provenance: the parity report next to it judged exactly this request
    (request_sha256), passed, came from a fake run, used the same passthrough route as this launch, and the fake
    run's bundle has the same identity (session, cut, sentinel, mode, hint, original request) as this bundle."""
    from .proxy.resume_proxy import request_digest, route_from_passthrough_args

    rec = json.loads(path.read_text())
    body = rec.get("body", rec)
    digest = request_digest(body)
    rep_path = path.with_name("first_request_parity.json")
    rep = json.loads(rep_path.read_text()) if rep_path.is_file() else {}
    problems = []
    if not rep.get("ok"):
        problems.append("parity report missing or not ok")
    if rep.get("request_sha256") != digest:
        problems.append("parity report did not judge this request")
    if rep.get("upstream_fake") is not True:
        problems.append("not a --fake run")
    if (rep.get("route") or {}) != route_from_passthrough_args(passthrough_args or []):
        problems.append(f"fake run route {rep.get('route')} differs from this launch's passthrough args")
    fake_bundle = path.parent.parent.parent / "resume" / "bundle.json"  # <trial>/agent/resume/bundle.json
    fake_meta = json.loads(fake_bundle.read_text()) if fake_bundle.is_file() else {}
    diff = [k for k in FREEZE_IDENTITY_KEYS if fake_meta.get(k) != meta.get(k)]
    if diff:
        problems.append(f"fake run's bundle differs from this bundle in {diff}")
    if problems:
        raise SystemExit(f"--freeze-first-request {path}: " + "; ".join(problems))
    return digest


def build_job_config(*, adapter, bundle: Path, task_dir: Path, job_name: str, jobs_dir: Path,
                     snapshot_state: Path | None = None, snapshot_id: str | None = None, fake: bool = False,
                     verifier: bool = True, model: str | None = None, upstream_url: str | None = None,
                     extra_env: dict[str, str] | None = None, extra_kwargs: dict[str, Any] | None = None,
                     original_trial_dir: str | None = None, setup_timeout_s: float = 1200.0,
                     passthrough_args: list[str] | None = None, parity_gate: str = "auto",
                     max_model_requests: int | None = None,
                     freeze_first_request: str | None = None) -> dict[str, Any]:
    bundle, meta = load_bundle(bundle)
    bundle = bundle.resolve()
    task_dir = task_dir.resolve()
    if not (task_dir / "task.toml").is_file():
        raise SystemExit(f"{task_dir} is not a Harbor task dir (no task.toml)")
    env_kwargs: dict[str, Any] = {"owner": config.owner_tag()}
    if snapshot_state:
        source = json.loads(Path(snapshot_state).read_text())
        if snapshot_id and source.get("snapshot_id") != snapshot_id:
            raise SystemExit("--snapshot-id differs from the snapshot state file")
        if not source.get("snapshot_id") or not source.get("replay_gate_passed"):
            raise SystemExit("snapshot source has not passed the replay gates")
        if source.get("run_id") and source["run_id"] != meta["run_id"]:
            check_shared_snapshot(source, bundle)
        hashes = verifier_hashes(task_dir)
        env_kwargs.update(snapshot_id=source["snapshot_id"], expected_fingerprint=source["fingerprint_source"],
                          fingerprint_command=source.get("fingerprint_command"),
                          expected_auxiliary_hashes=source.get("auxiliary_hashes", {}))
        if hashes:
            env_kwargs["expected_verifier_hashes"] = hashes
    elif snapshot_id:
        print("warning: no --snapshot-state; the clone's task fingerprint will NOT be checked", file=sys.stderr)
        env_kwargs["snapshot_id"] = snapshot_id
    else:
        raise SystemExit("need --snapshot-state (recommended) or --snapshot-id")
    orig_env, orig_kwargs = original_agent_settings(original_trial_dir)
    agent_env = {**getattr(adapter, "default_env", {}), **orig_env, **(extra_env or {})}
    upstream_url = upstream_url or config.upstream_url()
    kwargs = {**orig_kwargs, **(extra_kwargs or {}), "version": meta.get("agent_version_orig") or adapter.version,
              "resume_bundle": str(bundle), "upstream_fake": bool(fake), "parity_gate": parity_gate}
    if parity_gate == "require" and not meta.get("orig_request_at_k"):
        raise SystemExit("--parity-gate require: bundle has no orig_request_at_k "
                         f"({meta.get('orig_request_locate_error')})")
    if freeze_first_request:
        kwargs["expected_first_request_sha256"] = frozen_first_request_digest(
            Path(freeze_first_request), meta, passthrough_args)
    elif parity_gate == "require" and not fake:
        raise SystemExit("--parity-gate require for a real run needs --freeze-first-request <first_request_rewritten.json "
                         "of a --fake run of this bundle that passed the parity gate>")
    if max_model_requests is not None:
        if max_model_requests < 1:
            raise SystemExit("--max-model-requests must be >= 1")
        kwargs["max_model_requests"] = int(max_model_requests)
    if passthrough_args:
        from .proxy.runtime import validate_passthrough_args
        kwargs["passthrough_args"] = validate_passthrough_args(passthrough_args)
    agent: dict[str, Any] = {"import_path": adapter.harbor_import_path,
                             "model_name": model or config.model_override() or meta.get("model"),
                             "override_timeout_sec": float(meta["agent_budget_s"]),
                             "override_setup_timeout_sec": float(setup_timeout_s), "kwargs": kwargs, "env": agent_env}
    if not fake:
        if not upstream_url:
            raise SystemExit("set ASR_UPSTREAM_URL (or --upstream-url) for a real run, or use --fake")
        kwargs["upstream_url"] = upstream_url
        host = urlparse(upstream_url).hostname
        if host:
            agent["extra_allowed_hosts"] = [host]
    cfg: dict[str, Any] = {
        "job_name": job_name, "jobs_dir": str(jobs_dir.resolve()), "n_concurrent_trials": 1,
        "retry": {"max_retries": 0}, "agent_timeout_multiplier": 1.0,
        "environment": {"import_path": SNAPSHOT_ENV_IMPORT, "kwargs": env_kwargs},
        "agents": [agent], "tasks": [{"path": str(task_dir)}]}
    if not verifier:
        cfg["verifier"] = {"disable": True}
    return cfg


def validate_job_config(cfg: dict[str, Any]) -> None:
    """Parse with Harbor's own JobConfig model (raises on an invalid config)."""
    from harbor.models.job.config import JobConfig

    JobConfig.model_validate(cfg)


def run_harbor(config_path: Path, fake: bool) -> int:
    if not fake:
        missing = [k for k in ("ASR_UPSTREAM_KEY",) if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"missing in the environment: {missing} (never put it in the job config)")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # Harbor's Claude Code agent resolves a key/base URL at init; give it the dummy values the sandbox will use.
    env.update(ANTHROPIC_API_KEY="dummy-resume", ANTHROPIC_BASE_URL="http://127.0.0.1:4031")
    harbor = shutil.which("harbor") or str(Path(sys.executable).with_name("harbor"))
    cmd = [harbor, "run", "--config", str(config_path), "--yes"]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]
    return subprocess.call(cmd, env=env)
