"""Build a resume bundle from a Harbor trial dir and a step k (host side, read-only on the trial).

Bundle layout (<out>/<run_id>/):
  bundle.json            metadata: trial, session id, project slug, cwd, cut_line, agent budget, sentinel, ...
  session_trunc.jsonl    the native session's lines[:cut], byte-exact (ends right before step k)
  plan.json              replay plan: side-effecting actions before step k with their original observations
  aux/...                adapter-specific auxiliary files (Claude Code: aux/tool-results/*, aux/tasks/*.json)
  proxy_spec.json        resume-proxy spec (no secrets): mode, sentinel, hint, pin_date, resume artifacts

The remaining agent budget is  agent_timeout - elapsed(k),  where agent_timeout is the task's agent timeout
(--agent-timeout, else [agent].timeout_sec of --task-dir/task.toml, else of <trial>/task.toml).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

from .trace import ir as trace_ir

SENTINEL_PREFIX = "ZZZ_RESUME_SENTINEL_"


def resolve_agent_timeout(trial_dir: Path, agent_timeout: float | None = None,
                          task_dir: Path | None = None) -> float:
    if agent_timeout:
        return float(agent_timeout)
    for cand in ([task_dir / "task.toml"] if task_dir else []) + [trial_dir / "task.toml",
                                                                  trial_dir / "config" / "task.toml"]:
        if cand.is_file():
            t = ((tomllib.loads(cand.read_text()).get("agent") or {}).get("timeout_sec"))
            if t:
                return float(t)
    raise SystemExit("agent timeout unknown: pass --agent-timeout SECONDS or --task-dir <task with task.toml>")


def _trial_result(trial_dir: Path) -> dict[str, Any]:
    p = trial_dir / "result.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def default_run_id(trial: str, atif_step: int, mode: str) -> str:
    rid = f"{trial.split('__')[0][:28]}-{trial.split('__')[-1][:8]}-s{atif_step}-{mode}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", rid)


def sentinel_for(run_id: str) -> str:
    return SENTINEL_PREFIX + hashlib.sha1(run_id.encode()).hexdigest()[:10]


def build(*, adapter, trial_dir: Path, atif_step: int, out: Path, mode: str = "same", hint: str | None = None,
          agent_timeout: float | None = None, task_dir: Path | None = None, run_id: str | None = None,
          model: str | None = None, pin_date: bool = True, fingerprint_roots: list[str] | None = None,
          force: bool = False) -> tuple[Path, dict[str, Any]]:
    if mode not in ("same", "hint"):
        raise SystemExit("--mode must be same or hint")
    if mode == "hint" and not hint:
        raise SystemExit("--mode hint needs --hint-file")
    trial_dir = Path(trial_dir).resolve()
    trial = trial_dir.name
    result = _trial_result(trial_dir)
    task = (result.get("task_name") or "").split("/")[-1] or (task_dir.name if task_dir else trial.split("__")[0])
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    timeout = resolve_agent_timeout(trial_dir, agent_timeout, task_dir)
    ir = adapter.load_trace(trial_dir, agent_timeout_s=timeout, task=task, reward=reward)
    if ir.get("error"):
        raise SystemExit(f"{trial}: {ir['error']}")
    k = trace_ir.step_index(ir, atif_step)
    st = ir["steps"][k]
    if st["source"] != "agent":
        raise SystemExit("resume step must be an agent step")
    prev = ir["steps"][k - 1]
    sess_path = trial_dir / ir["sources"]["session_jsonl"]
    with open(sess_path, encoding="utf-8") as f:
        raw_lines = [l for l in f.read().split("\n") if l.strip()]
    cut = adapter.cut_line(raw_lines, ir, k)
    plan = trace_ir.replay_plan(ir, k)
    # attach original observation text for fidelity comparison
    obs: dict[str, Any] = {}
    for s in ir["steps"][:k]:
        for e in s["events"]:
            if e["type"] == "observation":
                obs[e["call_id"]] = e["text"]
    for sc in ir["sidechains"]:
        for c in sc["tool_calls"]:
            obs.setdefault(c["call_id"], c.get("observation_text"))
    for a in plan["actions"]:
        a["orig_text"] = obs.get(a["call_id"])
    first_cwd = next((json.loads(trace_ir._REDACT_RE.sub(r"\1null", l)).get("cwd")
                      for l in raw_lines if '"cwd"' in l), None)
    plan["initial_cwd"] = first_cwd
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id or default_run_id(trial, atif_step, mode))
    bdir = Path(out) / run_id
    if bdir.exists():
        if not force:
            raise SystemExit(f"{bdir} exists (use --force to overwrite)")
        shutil.rmtree(bdir)
    (bdir / "aux").mkdir(parents=True)
    trunc = "\n".join(raw_lines[:cut]) + "\n"
    with open(bdir / "session_trunc.jsonl", "w", encoding="utf-8") as f:
        f.write(trunc)
    with open(bdir / "plan.json", "w") as f:
        json.dump(plan, f, ensure_ascii=False)
    aux_counts = adapter.collect_aux(sess_path, ir, k, trunc, bdir / "aux")
    rp = next(r for r in trace_ir.resume_points(ir) if r["index"] == k)
    budget = round(timeout - (st["elapsed_s"] or 0), 1)
    if budget <= 0:
        raise SystemExit(f"no agent budget left at step {atif_step} (elapsed {st['elapsed_s']} s >= {timeout} s)")
    orig_date = (ir["start_ts"] or "")[:10]
    sentinel = sentinel_for(run_id)
    agent_model = model or ((result.get("config") or {}).get("agent") or {}).get("model_name") or ir.get("model")
    meta = {
        "run_id": run_id, "agent": adapter.name, "trial": trial, "trial_dir": str(trial_dir), "task": task,
        "task_path": str(task_dir.resolve()) if task_dir else None, "orig_reward": reward,
        "session_id": ir["session_id"], "project_slug": ir["project_slug"], "cwd": first_cwd,
        "agent_version_orig": ir.get("cc_version"), "resume_atif_step": atif_step, "resume_index": k,
        "prev_atif_step": prev["atif_step_id"], "cut_line": cut, "n_session_lines": len(raw_lines),
        "session_sha256": ir["sources"]["session_sha256"], "elapsed_at_k_s": st["elapsed_s"],
        "agent_timeout_task_s": timeout, "agent_budget_s": budget, "orig_date": orig_date, "mode": mode,
        "hint": hint, "model": agent_model, "sentinel": sentinel, "aux_counts": aux_counts,
        "resume_cost": rp, "n_replay_actions": len(plan["actions"]), "flags": ir["flags"],
        "fingerprint_roots": fingerprint_roots or None,
        "replay_reviewed_exceptions": {},
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "orig_step_k_preview": [{"type": e["type"], "tool": e.get("tool"),
                                 "text": (e.get("text") or json.dumps(e.get("args", ""), ensure_ascii=False))[:400]}
                                for e in st["events"] if e["type"] in ("assistant_reasoning", "assistant_text",
                                                                       "tool_call")],
    }
    with open(bdir / "bundle.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    spec = {"mode": mode, "sentinel": sentinel, "hint": hint if mode == "hint" else None,
            "upstream": "http://127.0.0.1:4013", "model": None, "pin_date": orig_date if pin_date else None,
            "resume_artifacts": dict(adapter.resume_artifacts)}
    with open(bdir / "proxy_spec.json", "w") as f:
        json.dump(spec, f, indent=1)
    return bdir, meta
