"""Agent-agnostic trajectory IR helpers.

The IR is a plain dict produced by an agent adapter's trace loader (see ``agents/base.py``). Every adapter must
emit main-chain ``steps`` with ``events`` and optional ``sidechains`` in the shape documented in
``docs/design.md``; everything in this module only reads that shape.

  step_index(ir, atif_step_id) -> int        IR index of a Harbor ATIF step id
  replay_plan(ir, k) -> dict                 ordered side-effecting actions needed to rebuild state before step k
  resume_points(ir) -> list[dict]            per-step resume cost summary
  bash_is_read_only(cmd) -> bool             conservative read-only shell heuristic
No model-specific chat template is rendered anywhere in this package.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shlex
from collections import Counter

IR_VERSION = "asr-trace-ir/0.1"
# Some export pipelines replace secret-looking values with a bare [REDACTED] token, which is not valid
# JSON. Such lines are parsed by substituting null; the raw bytes are kept unchanged for resume.
_REDACT_RE = re.compile(r"(?<=[:\[,])(\s*)\[REDACTED\](?=\s*[,}\]])")

SIDE_EFFECT_CLASSES = ["read_only", "file_write", "exec", "exec_background", "subagent", "task_mgmt", "web", "other"]

# conservative read-only Bash heuristic: every pipeline segment starts with a whitelisted command,
# no output redirection to files, no in-place flags, no command substitution writing files.
_RO_CMDS = {"ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "pwd", "echo", "printf",
            "which", "file", "stat", "du", "df", "nproc", "free", "uname", "whoami", "id", "env", "printenv",
            "date", "sort", "uniq", "cut", "tr", "awk", "column", "diff", "cmp", "md5sum", "sha256sum", "tree",
            "less", "more", "xxd", "od", "hexdump", "realpath", "dirname", "basename", "true", "nvidia-smi",
            "ps", "top", "uptime", "lscpu", "type", "command", "test", "[", "jq", "readlink", "sleep", "cd",
            "zcat", "gzip", "unzip", "tar", "h5dump", "ncdump", "strings", "nl", "seq", "lsof", "pgrep"}
_RO_ARG_BLOCK = re.compile(r"(^|\s)(-delete|-exec|-execdir|-i\b|--in-place|-x\b|-c\b.*>|>)")


def bash_is_read_only(cmd: str) -> bool:
    if not cmd or "<<" in cmd or re.search(r"(?<![<>&0-9])>(?!&)|>>|\btee\b", cmd.replace("2>&1", "").replace(">/dev/null", "").replace("> /dev/null", "").replace("2>/dev/null", "")):
        return False
    segs = re.split(r"\|\||&&|;|\||\n", cmd)
    for s in segs:
        s = s.strip()
        if not s:
            continue
        s = re.sub(r"^(\w+=\S+\s+)+", "", s)  # env assignments
        try:
            tok = shlex.split(s)[0]
        except (ValueError, IndexError):
            return False
        tok = os.path.basename(tok)
        if tok not in _RO_CMDS:
            return False
        if tok in ("tar", "unzip", "gzip") and not re.search(r"\s-(t|l|tf|tzf|tvf)\b|--list|\s-l\b|\s-c\b", s):
            return False
        if tok == "find" and re.search(r"-delete|-exec|-execdir|-fprint", s):
            return False
        if tok == "awk" and re.search(r"system\(|print\s*>", s):
            return False
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_lenient(path):
    txt = open(path, encoding="utf-8").read()
    try:
        return json.loads(txt), 0
    except json.JSONDecodeError:
        return json.loads(_REDACT_RE.sub(r"\1null", txt)), len(_REDACT_RE.findall(txt))


def _ts(s):
    if not s:
        return None
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def step_index(ir, atif_step_id):
    for s in ir["steps"]:
        if s["atif_step_id"] == atif_step_id:
            return s["index"]
    raise KeyError(f"atif step {atif_step_id} not a main-chain step of {ir['trial']}")


def replay_plan(ir, k_index, include_read_only=False):
    """Ordered actions needed to rebuild workspace state before main-chain step index k (exclusive).
    Sidechain (subagent) calls are merged by timestamp when their result landed before step k started."""
    k = k_index
    if k <= 0 or k >= len(ir["steps"]) or ir["steps"][k]["source"] != "agent":
        raise ValueError("resume step must be an agent step after the instruction")
    cut_ts = _ts(ir["steps"][k]["timestamp"])
    actions = []
    for st in ir["steps"][:k]:
        for e in st["events"]:
            if e["type"] != "tool_call":
                continue
            if e["side_effect"] == "read_only" and not include_read_only:
                continue
            actions.append({"origin": "main", "atif_step_id": st["atif_step_id"], "step_index": st["index"],
                            "call_id": e["call_id"], "tool": e["tool"], "side_effect": e["side_effect"],
                            "args": e["args"], "timestamp": e["timestamp"], "duration_s": e.get("duration_s"),
                            "is_error": e.get("is_error"), "background_task_id": e.get("background_task_id"),
                            "background_output_file": e.get("background_output_file"), "cwd": e.get("cwd"),
                            "auto_backgrounded": e.get("auto_backgrounded", False)})
    for sc in ir.get("sidechains", []):
        for c in sc["tool_calls"]:
            if c["side_effect"] == "read_only" and not include_read_only:
                continue
            if c.get("result_timestamp") and _ts(c["result_timestamp"]) < cut_ts:
                actions.append({"origin": f"sidechain:{sc['agent_id']}", "atif_step_id": None, "step_index": None,
                                "call_id": c["call_id"], "tool": c["tool"], "side_effect": c["side_effect"],
                                "args": c["args"], "timestamp": c["timestamp"], "duration_s": c.get("duration_s"),
                                "is_error": c.get("is_error"), "background_task_id": c.get("background_task_id"),
                                "cwd": c.get("cwd")})
    actions.sort(key=lambda a: (_ts(a["timestamp"]) or 0))
    # background task completions observed (task-notification) before step k
    done = {}
    for st in ir["steps"][:k]:
        for e in st["events"]:
            if e["type"] == "attachment" and e.get("bg_task_id"):
                done[e["bg_task_id"]] = {"status": e.get("bg_status"), "timestamp": e.get("timestamp")}
    for a in actions:
        if a.get("background_task_id"):
            info = done.get(a["background_task_id"])
            a["bg_finished_before_k"] = bool(info)
            a["bg_status"] = info["status"] if info else None
            a["bg_finished_ts"] = info["timestamp"] if info else None
            if info and a.get("timestamp"):
                a["bg_duration_s"] = round(_ts(info["timestamp"]) - _ts(a["timestamp"]), 1)
    return {"trial": ir["trial"], "resume_step_atif": ir["steps"][k]["atif_step_id"], "resume_step_index": k,
            "actions": actions}


def resume_points(ir):
    """Per agent step k: cost of rebuilding the prefix (steps < k).
    replay_exec_s = sum of original durations of foreground exec calls + finished background tasks (their
    original runtime until the task-notification) -> lower bound of replay wall time.
    bg_inflight / subagents_inflight > 0 means state at k contains running processes that a replay cannot
    reproduce exactly (avoid such k when possible)."""
    out = []
    acc = Counter()
    exec_s = 0.0
    bg_start = {}
    bg_done = set()
    sub_spawned = {}
    sub_done = set()
    side_ends = {sc.get("spawn_call_id"): sc.get("last_ts") for sc in ir.get("sidechains", [])}
    for st in ir["steps"]:
        if st["source"] == "agent" and st["index"] > 0:
            now = _ts(st["timestamp"])
            infl_sub = sum(1 for c, t in sub_spawned.items() if c not in sub_done and (side_ends.get(c) is None or _ts(side_ends[c]) > now))
            out.append({"atif_step_id": st["atif_step_id"], "index": st["index"], "elapsed_s": st["elapsed_s"],
                        "remaining_s": st["remaining_s"], "replay_exec_s": round(exec_s, 1),
                        "n_file_write": acc["file_write"], "n_exec": acc["exec"],
                        "n_exec_background": acc["exec_background"], "n_subagent": len(sub_spawned),
                        "bg_inflight": len(set(bg_start) - bg_done), "subagents_inflight": infl_sub,
                        "n_task_mgmt": acc["task_mgmt"], "n_web": acc["web"], "n_read_only": acc["read_only"],
                        "context_tokens_prev": _prev_ctx(ir, st["index"])})
        for e in st["events"]:
            if e["type"] == "tool_call":
                acc[e["side_effect"]] += 1
                if e["side_effect"] == "exec" and e.get("duration_s"):
                    exec_s += max(0.0, e["duration_s"])
                if e["side_effect"] == "subagent" and e["tool"] in ("Agent", "Task"):
                    sub_spawned[e["call_id"]] = e["timestamp"]
                if e.get("background_task_id"):
                    bg_start[e["background_task_id"]] = e["timestamp"]
            elif e["type"] == "attachment" and e.get("bg_task_id"):
                tid = e["bg_task_id"]
                if tid in bg_start and tid not in bg_done:
                    exec_s += max(0.0, (_ts(e["timestamp"]) or 0) - (_ts(bg_start[tid]) or 0))
                bg_done.add(tid)
                if e.get("bg_call_id") in sub_spawned:
                    sub_done.add(e["bg_call_id"])
            elif e["type"] == "observation" and e.get("tool") in ("Agent", "Task") and "async_launched" not in e["text"][:300] and "Async agent launched" not in e["text"][:200]:
                sub_done.add(e["call_id"])
            elif e["type"] == "observation" and e.get("tool") == "TaskStop":
                for tid in list(bg_start):
                    if tid in e["text"]:
                        bg_done.add(tid)
    return out


def _prev_ctx(ir, k):
    for s in reversed(ir["steps"][:k + 1]):
        if s.get("context_tokens"):
            return s["context_tokens"]
    return None
