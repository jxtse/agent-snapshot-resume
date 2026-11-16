"""Regrade a finished trial: run its verifier against the recorded artifacts, never the agent again.

Harbor's regrade trial replaces the agent phase with "restore the recorded outputs", so a trial whose agent work
finished but whose verification never ran (or ran against a broken verifier image) can be graded from what it already
produced. This module builds that trial with *this* project's environment class, because the Aliyun FC verifier
sandbox is not Harbor's default docker environment and it has to carry the same gates a live launch does
(``expected_verifier_hashes``: the verifier's /tests must be the task's tests/, restored from the task dir when the
image ships an incomplete copy).

The source trial is read, never modified: agent logs and artifacts are copied into a fresh trial directory.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .launch import SNAPSHOT_ENV_IMPORT, verifier_hashes


def build_regrade_config(*, source: Path, task_dir: Path, trials_dir: Path, owner: str, job: str,
                         trial_name: str | None = None) -> Any:
    """The harbor TrialConfig of a regrade trial of `source` with this project's verifier environment."""
    from harbor.models.trial.config import (EnvironmentConfig, SourceTrialConfig, TaskConfig, TrialConfig,
                                            VerifierConfig)
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import TrialResult
    from harbor.trial.regrade import check_task_regradable

    source = Path(source).resolve()
    task_dir = Path(task_dir).resolve()
    paths = TrialPaths(trial_dir=source)
    if not paths.result_path.is_file():
        raise SystemExit(f"{source} has no result.json; nothing to regrade")
    result = TrialResult.model_validate_json(paths.result_path.read_text())
    reason = check_task_regradable(task_dir)
    if reason:
        raise SystemExit(reason)

    agent, artifacts = result.config.agent, result.config.artifacts
    if paths.config_path.is_file():  # the recorded run config is the more complete record when it survives
        try:
            recorded = json.loads(paths.config_path.read_text())
            agent = recorded.get("agent") or agent
            artifacts = recorded.get("artifacts") or artifacts
        except ValueError:
            pass

    env_kwargs: dict[str, Any] = {"owner": owner, "job": job}
    hashes = verifier_hashes(task_dir)
    if hashes:
        env_kwargs["expected_verifier_hashes"] = hashes
    return TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name=trial_name or f"regrade-{source.name}",
        trials_dir=Path(trials_dir),
        agent=agent,
        environment=EnvironmentConfig(import_path=SNAPSHOT_ENV_IMPORT, kwargs=env_kwargs),
        verifier=VerifierConfig(),
        artifacts=artifacts,
        source_trial=SourceTrialConfig(action="regrade", type="local", trial_id=result.id, path=source),
    )


def regrade(*, source: Path, task_dir: Path, trials_dir: Path, owner: str, job: str,
            trial_name: str | None = None) -> dict[str, Any]:
    """Run the regrade trial and return its verifier result (plus where it was written)."""
    from harbor.trial.trial import Trial

    config = build_regrade_config(source=source, task_dir=task_dir, trials_dir=trials_dir, owner=owner, job=job,
                                  trial_name=trial_name)

    async def _run():
        trial = await Trial.create(config)
        return await trial.run()

    result = asyncio.run(_run())
    out: dict[str, Any] = {"source": str(Path(source).resolve()), "trial_dir": str(Path(trials_dir) / config.trial_name),
                           "rewards": (result.verifier_result.rewards if result.verifier_result else None),
                           "exception": (result.exception_info.model_dump() if result.exception_info else None)}
    (Path(trials_dir) / config.trial_name / "regrade.json").write_text(json.dumps(out, indent=1))
    return out
