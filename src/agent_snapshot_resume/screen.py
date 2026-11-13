"""Offline replayability screen for a candidate resume cut (no sandbox, no model call).

`asr screen --trial-dir T --cut-line L [--cut-line L2 ...]` (or `--step`) builds the replay plan for each cut and
reports what may make the source replay diverge from the original prefix, so cuts can be chosen (or dropped) before
paying for a sandbox. Every risk names the plan action (index, call id, step) and the evidence. Severity:

  block   the replay gate is expected to fail, or the restored state is known to be different
  warn    the gate may fail (pace-, order-, or environment-dependent output); prepare-snapshot will tell
  info    worth knowing (long replay, compactions before the cut, ...)

Rules (generic; see `RULES`):
  bg_inflight            a background job still running at the cut (state contains a live process; replay restarts
                         it detached, so the agent sees a different progress point)
  subagent_inflight      a sub-agent still running at the cut
  pace_dependent_read    a command reads a log/output of an earlier background job, or polls with sleep/ps/tail
                         while one is running: its output depended on wall-clock pacing
  unseeded_rng           numpy/random/torch RNG used without a seed in a command that prints numbers
  network                downloads / package installs / remote API calls (network may differ or be offline)
  glob_order             glob/listdir/os.walk/iterdir/find without sorting (filesystem order differs per sandbox)
  agent_state_read       reads the agent's own logs/session files (not present, or different, during replay)
  wall_clock             prints dates/times/durations (accepted automatically only for labelled durations)
  hash_order             iterates a Python set or uses hash() of str (PYTHONHASHSEED differs per process)
  web_or_unsupported     WebFetch/WebSearch, NotebookEdit or unknown tools in the prefix (not replayed)
  orig_timeout           a command hit Claude Code's timeout originally (replay reproduces the cut-off)
  long_replay            sum of original exec durations (lower bound of the replay wall time)
  compactions            context compactions before the cut (needs the post-compaction parity gate)
  missing_original       original request at the cut not located in the proxy log (parity gate unavailable)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .trace import ir as trace_ir

BLOCK, WARN, INFO = "block", "warn", "info"

_RNG_USE = re.compile(r"np\.random\.(?:rand|randn|randint|normal|uniform|choice|permutation|shuffle|default_rng\(\s*\))|"
                      r"\brandom\.(?:random|randint|choice|shuffle|uniform|gauss|sample)\(|torch\.(?:rand|randn)\(|"
                      r"default_rng\(\s*\)")
_SEED = re.compile(r"seed\s*\(|default_rng\(\s*\w|random_state\s*=|RandomState\(\s*\d|manual_seed|PYTHONHASHSEED")
_NET = re.compile(r"\b(?:curl|wget|git\s+clone|pip3?\s+install|uv\s+(?:pip|add|sync)|apt(?:-get)?\s+install|"
                  r"conda\s+install|npm\s+(?:i|install)|hf\s+download|huggingface-cli)\b|urllib\.request|"
                  r"requests\.(?:get|post)|urlopen\(|http[s]?://")
_LISTING = re.compile(r"\b(?:glob\.i?glob|os\.listdir|os\.scandir|os\.walk)\(|\.(?:r?glob|iterdir)\(")
_FIND_UNSORTED = re.compile(r"\bfind\s+\S+[^|;&\n]*(?:\|\s*(?!sort)\w+|$)", re.M)
_CLOCK = re.compile(r"time\.time\(|perf_counter|datetime\.(?:datetime\.)?now\(|date\.today\(|timeit|"
                    r"(?:^|[;&|]\s*)date(?:\s|$)", re.M)
_AGENT_STATE = re.compile(r"/logs/agent\b|\.claude/|CLAUDE_CONFIG_DIR|/sessions/projects/")
_HASH = re.compile(r"\bhash\(\s*['\"]|for\s+\w+\s+in\s+set\(|print\(\s*set\(|\bset\([^)]*\)\s*\)")
_POLL = re.compile(r"\bsleep\s+\d|\bps\s+(?:aux|-ef)|\bpgrep\b|\bjobs\b|\btail\s+(?:-n\s*\d+\s+)?\S+\.(?:log|out|txt)\b|"
                   r"\bwc\s+-l\s+\S+\.(?:log|csv)\b|nvidia-smi")
_BG_OUT = re.compile(r"(?:>|>>)\s*(\S+)\s*(?:2>&1)?\s*&\s*$|nohup\s+.*?(?:>|>>)\s*(\S+)", re.M)

RULES = ("bg_inflight", "subagent_inflight", "pace_dependent_read", "unseeded_rng", "network", "glob_order",
         "agent_state_read",
         "wall_clock", "hash_order", "web_or_unsupported", "orig_timeout", "long_replay", "compactions",
         "missing_original")


def _cmd(a: dict[str, Any]) -> str:
    return (a.get("args") or {}).get("command", "") or ""


def _unsorted_listing(cmd: str) -> bool:
    """A directory listing whose order reaches the output: glob/listdir/walk/iterdir calls not wrapped in sorted(...)
    or `find` piped into something other than sort."""
    for m in _LISTING.finditer(cmd):
        head = cmd[max(0, m.start() - 40):m.start()]
        if re.search(r"sorted\(\s*[\w.]*$|sorted\([^()]*$", head):
            continue
        return True
    return bool(_FIND_UNSORTED.search(cmd)) and "sort" not in cmd


def _python_like(cmd: str) -> bool:
    return bool(re.search(r"\bpython3?\b|\.py\b|<<\s*'?EOF", cmd))


def screen_plan(plan: dict[str, Any], rp: dict[str, Any] | None = None, *, n_compactions: int = 0,
                orig_request: dict[str, Any] | None = None, long_replay_s: float = 1800.0) -> dict[str, Any]:
    risks: list[dict[str, Any]] = []

    def add(rule, sev, a=None, i=None, **kw):
        r = {"rule": rule, "severity": sev}
        if a is not None:
            r.update(action=i, call_id=a.get("call_id"), atif_step_id=a.get("atif_step_id"),
                     command=_cmd(a)[:200] or None)
        r.update(kw)
        risks.append(r)

    bg_outputs: dict[str, dict[str, Any]] = {}  # file path -> background action
    bg_running: list[dict[str, Any]] = []
    for i, a in enumerate(plan["actions"]):
        cmd = _cmd(a)
        se = a.get("side_effect")
        ts = trace_ir._ts(a.get("timestamp"))
        # background jobs alive at this action (by original timestamps)
        alive = [b for b in bg_running if (b.get("_end") is None or (ts is not None and b["_end"] > ts))]
        if se == "exec_background":
            if not a.get("bg_finished_before_k", True):
                add("bg_inflight", BLOCK, a, i, detail="background job still running at the cut")
            end = trace_ir._ts(a.get("bg_finished_ts")) if a.get("bg_finished_ts") else None
            bg_running.append(dict(a, _end=end))
            if a.get("background_output_file"):
                bg_outputs[a["background_output_file"]] = a
        elif se == "exec" and re.search(r"&\s*(?:$|\n)|\bnohup\b|\bsetsid\b|\bdisown\b", cmd):
            # shell-level background job started by a foreground command: its end time is unknown
            for m in _BG_OUT.finditer(cmd):
                path = m.group(1) or m.group(2)
                if path:
                    bg_outputs[path] = a
            bg_running.append(dict(a, _end=None, _shell_bg=True))
            add("pace_dependent_read", WARN, a, i, detail="starts a shell background job (nohup/&); later reads of "
                "its output depend on how far it got (replay runs back-to-back)")
        if se in ("exec",) and i > 0:
            reads = [p for p in bg_outputs if p and p in cmd and bg_outputs[p] is not a]
            if reads:
                src = bg_outputs[reads[0]]
                end = trace_ir._ts(src.get("bg_finished_ts")) if src.get("bg_finished_ts") else None
                if src.get("side_effect") == "exec_background" and end is not None and ts is not None and end <= ts:
                    # Claude Code-managed job that had finished when it was read: the replay runs such jobs in the
                    # foreground to completion, so the read sees the same finished output
                    add("pace_dependent_read", INFO, a, i, detail=f"reads finished output of background job "
                        f"{src.get('call_id')}", background_finished_before_read=True)
                else:
                    add("pace_dependent_read", BLOCK if src.get("side_effect") == "exec_background" else WARN, a, i,
                        detail=f"reads output of background job {src.get('call_id')} ({reads[0]}) while it was "
                        "still running (or with unknown end): output depends on progress",
                        background_finished_before_read=False)
            elif alive and _POLL.search(cmd):
                add("pace_dependent_read", WARN, a, i, detail="polls (sleep/ps/tail/wc) while a background job "
                    "was running in the original run")
        if se in ("exec", "exec_background"):
            orig = a.get("orig_text") or ""
            if _RNG_USE.search(cmd) and not _SEED.search(cmd) and re.search(r"\d", orig):
                add("unseeded_rng", WARN, a, i)
            if _NET.search(cmd):
                add("network", WARN, a, i)
            if _unsorted_listing(cmd):
                add("glob_order", WARN, a, i)
            if _AGENT_STATE.search(cmd):
                add("agent_state_read", WARN, a, i, detail="reads the agent's own logs/session (absent or different "
                    "during replay)")
            if _CLOCK.search(cmd):
                add("wall_clock", INFO, a, i)
            if _python_like(cmd) and _HASH.search(cmd):
                add("hash_order", WARN, a, i)
            if orig.startswith("Exit code 143\nCommand timed out"):
                add("orig_timeout", INFO, a, i)
        if se == "web" or a.get("tool") in ("NotebookEdit",) or se == "other":
            add("web_or_unsupported", BLOCK if se == "other" or a.get("tool") == "NotebookEdit" else WARN, a, i)
    if rp:
        if rp.get("subagents_inflight"):
            add("subagent_inflight", BLOCK, detail=f"{rp['subagents_inflight']} sub-agent(s) running at the cut")
        if (rp.get("replay_exec_s") or 0) > long_replay_s:
            add("long_replay", INFO, replay_exec_s=rp["replay_exec_s"])
    if n_compactions:
        add("compactions", INFO, n=n_compactions)
    if orig_request is None:
        add("missing_original", WARN, detail="original request at the cut not located; parity gate unavailable")
    by = {s: sum(r["severity"] == s for r in risks) for s in (BLOCK, WARN, INFO)}
    verdict = "block" if by[BLOCK] else ("review" if by[WARN] else "ok")
    return {"verdict": verdict, "counts": by, "n_actions": len(plan["actions"]), "risks": risks}


def screen_trial(adapter, trial_dir: Path, *, cut_lines: list[int] | None = None, steps: list[int] | None = None,
                 agent_timeout_s: float | None = None, task_dir: Path | None = None) -> list[dict[str, Any]]:
    from . import parity
    from .bundle import plan_with_observations, resolve_agent_timeout, step_for_cut_line
    trial_dir = Path(trial_dir)
    ir = adapter.load_trace(trial_dir, agent_timeout_s=resolve_agent_timeout(trial_dir, agent_timeout_s, task_dir),
                            task=trial_dir.name.split("__")[0])
    if ir.get("error"):
        raise SystemExit(f"{trial_dir.name}: {ir['error']}")
    raw = [l for l in (trial_dir / ir["sources"]["session_jsonl"]).read_text(encoding="utf-8").split("\n")
           if l.strip()]
    reqs = parity.original_requests(trial_dir)
    rps = {r["index"]: r for r in trace_ir.resume_points(ir)}
    targets = list(steps or [])
    for cl in cut_lines or []:
        targets.append(step_for_cut_line(ir, int(cl), lambda i: adapter.cut_line(raw, ir, i)))
    out = []
    for sid in targets:
        k = trace_ir.step_index(ir, sid)
        plan = plan_with_observations(ir, k)
        n_comp = sum(1 for s in ir["steps"][:k] for e in s["events"] if e["type"] == "compaction_boundary")
        loc = parity.locate_request(ir, k, reqs)
        rep = screen_plan(plan, rps.get(k), n_compactions=n_comp, orig_request=loc.get("record"))
        rep.update(trial=trial_dir.name, atif_step=sid, cut_line=adapter.cut_line(raw, ir, k),
                   step_first_line=ir["steps"][k]["line_range"][0],
                   elapsed_s=ir["steps"][k].get("elapsed_s"), remaining_s=ir["steps"][k].get("remaining_s"),
                   replay_exec_s=(rps.get(k) or {}).get("replay_exec_s"), orig_request_index=loc.get("index"))
        out.append(rep)
    return out


def summarize(rep: dict[str, Any]) -> dict[str, Any]:
    rules: dict[str, int] = {}
    for r in rep["risks"]:
        key = f"{r['severity']}:{r['rule']}"
        rules[key] = rules.get(key, 0) + 1
    return {k: rep.get(k) for k in ("trial", "atif_step", "cut_line", "verdict", "n_actions", "replay_exec_s",
                                    "remaining_s", "orig_request_index")} | {"rules": rules}


def main_json(reports: list[dict[str, Any]], full: bool) -> str:
    return "\n".join(json.dumps(r if full else summarize(r), ensure_ascii=False) for r in reports)
