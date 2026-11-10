"""Claude Code session -> trajectory IR (validated against Claude Code 2.1.207 sessions written by Harbor).

Source of truth: the native session jsonl (``agent/sessions/projects/<slug>/<sid>.jsonl`` inside a Harbor trial dir,
plus ``<sid>/subagents/agent-*.jsonl`` sidechains and ``<sid>/tool-results/*`` persisted outputs). When Harbor's
ATIF ``agent/trajectory.json`` is present and aligned, every main-chain IR step carries its ATIF ``step_id``.

  load_trial(trial_dir, agent_timeout_s=None, task=None, reward=None) -> dict
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

from .ir import IR_VERSION, _REDACT_RE, _sha256, _ts, bash_is_read_only, read_json_lenient

SIDE_EFFECT = {
    "Read": "read_only", "Glob": "read_only", "Grep": "read_only", "LS": "read_only",
    "TaskOutput": "read_only", "TaskGet": "read_only", "TaskList": "read_only", "CronList": "read_only",
    "ToolSearch": "read_only", "Skill": "read_only", "ListAgents": "read_only",
    "Write": "file_write", "Edit": "file_write", "MultiEdit": "file_write", "NotebookEdit": "file_write",
    "Bash": "exec",  # refined per call
    "Agent": "subagent", "Task": "subagent", "SendMessage": "subagent",
    "TaskCreate": "task_mgmt", "TaskUpdate": "task_mgmt", "TaskStop": "task_mgmt", "TodoWrite": "task_mgmt",
    "CronCreate": "task_mgmt", "CronDelete": "task_mgmt", "ScheduleWakeup": "task_mgmt",
    "WebFetch": "web", "WebSearch": "web",
}


def read_session(path):
    """Return (records, raw_lines, n_redaction_repaired_lines). Raw lines are kept byte-exact (str) for resume."""
    recs, raws, repaired = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                d = json.loads(_REDACT_RE.sub(r"\1null", raw))
                repaired += 1
                d["_ir_repaired_redaction"] = True
            recs.append(d)
            raws.append(raw)
    return recs, raws, repaired



def find_session(trial_dir):
    ss = sorted(glob.glob(os.path.join(trial_dir, "agent/sessions/projects/*/*.jsonl")))
    return ss[0] if len(ss) == 1 else (None if not ss else ss)


def _content_text(c):
    """Flatten tool_result content to text; images -> placeholder (count returned separately)."""
    if c is None:
        return "", 0
    if isinstance(c, str):
        return c, 0
    parts, n_img = [], 0
    for b in c:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            n_img += 1
            parts.append("[image]")
        else:
            parts.append(json.dumps(b, ensure_ascii=False))
    return "\n".join(parts), n_img


_PERSIST_RE = re.compile(r"(?:saved to|Full output saved to|persisted to)[:\s]+(\S+/tool-results/\S+)", re.I)
_BG_RE = re.compile(r"Command running in background with ID: (\w+)\. Output is being written to: (\S+?)\.?(?:\s|$)")


def load_trial(trial_dir, agent_timeout_s=None, task=None, reward=None):
    if agent_timeout_s is None:
        raise ValueError("agent_timeout_s is required (the task's agent timeout in seconds)")
    trial = os.path.basename(trial_dir.rstrip("/"))
    sess = find_session(trial_dir)
    flags = []
    if sess is None:
        return {"ir_version": IR_VERSION, "trial": trial, "error": "no_session_jsonl", "steps": []}
    if isinstance(sess, list):
        return {"ir_version": IR_VERSION, "trial": trial, "error": f"multiple_sessions:{len(sess)}", "steps": []}
    recs, raws, repaired = read_session(sess)
    if repaired:
        flags.append(f"redaction_repaired_lines:{repaired}")
    n_redact_text = sum(r.count("[REDACTED") for r in raws)
    if n_redact_text:
        flags.append(f"redaction_tokens:{n_redact_text}")
    atif_path = os.path.join(trial_dir, "agent/trajectory.json")
    atif, atif_rep = (read_json_lenient(atif_path) if os.path.exists(atif_path) else (None, 0))
    if atif_rep:
        flags.append(f"atif_redaction_repaired:{atif_rep}")
    sid = os.path.basename(sess)[:-6]
    slug = os.path.basename(os.path.dirname(sess))
    sess_dir = os.path.join(os.path.dirname(sess), sid)
    cfg = json.load(open(os.path.join(trial_dir, "config.json"))) if os.path.exists(os.path.join(trial_dir, "config.json")) else {}
    cc_version = next((r.get("version") for r in recs if r.get("version")), None)
    timeout = float(agent_timeout_s)

    # tool_use index (main chain)
    calls = {}
    for i, d in enumerate(recs):
        if d.get("type") == "assistant":
            for b in d["message"]["content"]:
                if b.get("type") == "tool_use":
                    calls[b["id"]] = (i, b)

    # group lines into steps: a step starts at the first line of an assistant message id, or at a user line
    # carrying text (instruction / injected user text). Everything else attaches to the current step.
    steps, cur, seen_mid = [], None, set()
    for i, d in enumerate(recs):
        t = d.get("type")
        starts = False
        if t == "assistant":
            mid = d["message"].get("id")
            if mid not in seen_mid:
                seen_mid.add(mid)
                starts = True
        elif t == "user":
            c = d["message"]["content"]
            if isinstance(c, str) or any(b.get("type") == "text" for b in c):
                starts = True
        if starts:
            cur = {"source": "agent" if t == "assistant" else "user", "line_start": i, "lines": [i],
                   "timestamp": d.get("timestamp"), "message_id": d["message"].get("id") if t == "assistant" else None}
            steps.append(cur)
        elif cur is not None:
            cur["lines"].append(i)
        else:  # preamble (queue-operation etc.)
            pass
    for j, s in enumerate(steps):
        s["line_end"] = steps[j + 1]["line_start"] if j + 1 < len(steps) else len(recs)

    # ATIF alignment
    main_atif = [s for s in (atif or {}).get("steps", []) if not (s.get("extra") or {}).get("is_sidechain")]
    side_atif = [s for s in (atif or {}).get("steps", []) if (s.get("extra") or {}).get("is_sidechain")]
    aligned = bool(atif) and len(main_atif) == len(steps) and all(
        a["timestamp"] == s["timestamp"] for a, s in zip(main_atif, steps))
    if atif and not aligned:
        flags.append("atif_alignment_failed")
    if not atif:
        # No ATIF file: use the 1-based position of each main-chain step as its step id.
        flags.append("atif_missing_step_ids_positional")

    t0 = _ts(steps[0]["timestamp"]) if steps else None
    cum_in = cum_out = 0
    out_steps = []
    bg_tasks = {}
    for j, s in enumerate(steps):
        ev = []
        usage = None
        model = None
        stop_reason = None
        for i in range(s["line_start"], s["line_end"]):
            d = recs[i]
            t = d.get("type")
            if t == "assistant":
                m = d["message"]
                model = m.get("model") or model
                usage = m.get("usage") or usage
                stop_reason = m.get("stop_reason") or stop_reason
                for b in m["content"]:
                    bt = b.get("type")
                    if bt == "thinking":
                        ev.append({"type": "assistant_reasoning", "line": i, "text": b.get("thinking", "")})
                    elif bt == "redacted_thinking":
                        ev.append({"type": "assistant_reasoning", "line": i, "text": "", "redacted": True})
                    elif bt == "text":
                        if b.get("text", "") or not ev:
                            ev.append({"type": "assistant_text", "line": i, "text": b.get("text", "")})
                    elif bt == "tool_use":
                        name, args = b.get("name"), b.get("input") or {}
                        cls = SIDE_EFFECT.get(name, "other")
                        if name == "Bash":
                            if args.get("run_in_background"):
                                cls = "exec_background"
                            elif bash_is_read_only(args.get("command", "")):
                                cls = "read_only"
                        if name in ("Agent", "Task") and "subagent_type" not in args and "prompt" not in args:
                            cls = "other"
                        ev.append({"type": "tool_call", "line": i, "call_id": b["id"], "tool": name, "args": args,
                                   "side_effect": cls, "timestamp": d.get("timestamp"), "cwd": d.get("cwd")})
            elif t == "user":
                c = d["message"]["content"]
                if isinstance(c, str):
                    ev.append({"type": "instruction" if j == 0 else "user_text", "line": i, "text": c,
                               "is_meta": bool(d.get("isMeta"))})
                    continue
                for b in c:
                    bt = b.get("type")
                    if bt == "text":
                        ev.append({"type": "instruction" if j == 0 else "user_text", "line": i, "text": b.get("text", ""),
                                   "is_meta": bool(d.get("isMeta"))})
                    elif bt == "tool_result":
                        txt, n_img = _content_text(b.get("content"))
                        tur = d.get("toolUseResult")
                        o = {"type": "observation", "line": i, "call_id": b.get("tool_use_id"),
                             "is_error": bool(b.get("is_error")), "text": txt, "n_chars": len(txt),
                             "n_images": n_img, "timestamp": d.get("timestamp")}
                        if isinstance(tur, dict):
                            if tur.get("backgroundTaskId"):
                                o["background_task_id"] = tur["backgroundTaskId"]
                            if tur.get("interrupted"):
                                o["interrupted"] = True
                            if tur.get("persistedOutputPath"):
                                o["persisted_path"] = tur.get("persistedOutputPath")
                        mm = _PERSIST_RE.search(txt[:2000])
                        if mm and "persisted_path" not in o:
                            o["persisted_path"] = mm.group(1)
                        if "persisted_path" in o or "<persisted-output>" in txt[:200]:
                            o["truncated"] = True
                        bm = _BG_RE.search(txt[:600])
                        if bm:
                            o["background_task_id"] = bm.group(1)
                            o["background_output_file"] = bm.group(2)
                        ci = calls.get(o["call_id"])
                        if ci:
                            o["tool"] = ci[1].get("name")
                            cts = _ts(recs[ci[0]].get("timestamp"))
                            if cts and o["timestamp"]:
                                o["duration_s"] = round(_ts(o["timestamp"]) - cts, 3)
                        ev.append(o)
            elif t == "attachment":
                a = d.get("attachment") or {}
                ae = {"type": "attachment", "line": i, "subtype": a.get("type"),
                      "text": _attachment_text(a), "timestamp": d.get("timestamp")}
                if a.get("type") == "queued_command" and isinstance(a.get("prompt"), str) and "<task-notification>" in a["prompt"]:
                    tn = dict(re.findall(r"<(task-id|status|tool-use-id|output-file)>([^<]*)<", a["prompt"]))
                    ae.update({"bg_task_id": tn.get("task-id"), "bg_status": tn.get("status"),
                               "bg_call_id": tn.get("tool-use-id"), "bg_output_file": tn.get("output-file")})
                ev.append(ae)
            elif t == "system":
                st = d.get("subtype")
                ev.append({"type": "compaction_boundary" if st == "compact_boundary" else "system", "line": i,
                           "subtype": st, "text": (d.get("content") or "")[:2000] if isinstance(d.get("content"), str) else "",
                           "timestamp": d.get("timestamp")})
            elif t in ("last-prompt", "queue-operation", "summary", "file-history-snapshot"):
                if t == "queue-operation" and d.get("operation") == "enqueue":
                    ev.append({"type": "system", "line": i, "subtype": "queue_enqueue",
                               "text": (d.get("content") or "") if isinstance(d.get("content"), str) else "",
                               "timestamp": d.get("timestamp")})
            else:
                ev.append({"type": "system", "line": i, "subtype": f"unknown:{t}", "text": ""})
        # mark tool calls whose result says auto-backgrounded
        obs_by_call = {e["call_id"]: e for e in ev if e["type"] == "observation"}
        for e in ev:
            if e["type"] == "tool_call":
                o = obs_by_call.get(e["call_id"])
                if o is not None:
                    e["duration_s"] = o.get("duration_s")
                    e["is_error"] = o.get("is_error")
                    if o.get("background_task_id"):
                        e["background_task_id"] = o["background_task_id"]
                        e["background_output_file"] = o.get("background_output_file")
                        if e["side_effect"] in ("exec", "read_only") and e["tool"] == "Bash":
                            e["side_effect"] = "exec_background"
                            e["auto_backgrounded"] = True
                        bg_tasks[o["background_task_id"]] = {"call_id": e["call_id"], "step_index": j}
                else:
                    e["result_missing"] = True
        ctx = None
        if usage:
            ctx = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            cum_in += ctx
            cum_out += int(usage.get("output_tokens") or 0)
        ts = _ts(s["timestamp"])
        el = round(ts - t0, 3) if (ts is not None and t0 is not None) else None
        a = main_atif[j] if aligned else None
        out_steps.append({
            "atif_step_id": a["step_id"] if a else (None if atif else j + 1), "index": j, "source": s["source"],
            "kind": "instruction" if j == 0 else ("agent" if s["source"] == "agent" else "user_text"),
            "timestamp": s["timestamp"], "elapsed_s": el,
            "remaining_s": round(timeout - el, 3) if el is not None else None,
            "line_range": [s["line_start"], s["line_end"]], "message_id": s["message_id"], "model": model,
            "stop_reason": stop_reason,
            "usage": ({k: usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")} if usage else None),
            "context_tokens": ctx, "cum_prompt_tokens": cum_in if usage else None, "cum_output_tokens": cum_out if usage else None,
            "events": ev,
        })

    # sidechains (subagents)
    sidechains = []
    for p in sorted(glob.glob(os.path.join(sess_dir, "subagents", "agent-*.jsonl"))):
        srecs, _, srep = read_session(p)
        meta_p = p[:-6] + ".meta.json"
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        tools = Counter()
        side_calls = []
        sres = {}
        for i, d in enumerate(srecs):
            if d.get("type") == "user" and isinstance(d["message"]["content"], list):
                for b in d["message"]["content"]:
                    if b.get("type") == "tool_result":
                        txt, _ = _content_text(b.get("content"))
                        sres[b.get("tool_use_id")] = (d.get("timestamp"), bool(b.get("is_error")), txt, d.get("toolUseResult"))
        for i, d in enumerate(srecs):
            if d.get("type") == "assistant":
                for b in d["message"]["content"]:
                    if b.get("type") == "tool_use":
                        tools[b["name"]] += 1
                        name, args = b["name"], b.get("input") or {}
                        cls = SIDE_EFFECT.get(name, "other")
                        if name == "Bash":
                            cls = "exec_background" if args.get("run_in_background") else ("read_only" if bash_is_read_only(args.get("command", "")) else "exec")
                        r = sres.get(b["id"])
                        side_calls.append({"line": i, "call_id": b["id"], "tool": name, "args": args, "side_effect": cls,
                                           "timestamp": d.get("timestamp"), "cwd": d.get("cwd"), "result_timestamp": r[0] if r else None,
                                           "is_error": r[1] if r else None,
                                           "duration_s": round(_ts(r[0]) - _ts(d.get("timestamp")), 3) if r else None,
                                           "observation_text": r[2] if r else None,
                                           "background_task_id": (r[3] or {}).get("backgroundTaskId") if r and isinstance(r[3], dict) else None})
        sidechains.append({"agent_id": os.path.basename(p)[6:-6], "path": os.path.relpath(p, trial_dir),
                           "spawn_call_id": meta.get("toolUseId"), "agent_type": meta.get("agentType"),
                           "description": meta.get("description"), "n_lines": len(srecs), "redaction_repaired": srep,
                           "first_ts": srecs[0].get("timestamp") if srecs else None,
                           "last_ts": srecs[-1].get("timestamp") if srecs else None,
                           "tool_mix": dict(tools), "tool_calls": side_calls,
                           "atif_step_ids": [a["step_id"] for a in side_atif if (a.get("extra") or {}).get("agent_id") in (None, os.path.basename(p)[6:-6])][:0]})
    # attach sidechain refs to the spawning step
    by_call = {sc["spawn_call_id"]: sc for sc in sidechains if sc.get("spawn_call_id")}
    for st in out_steps:
        for e in list(st["events"]):
            if e["type"] == "tool_call" and e["call_id"] in by_call:
                sc = by_call[e["call_id"]]
                st["events"].append({"type": "sidechain", "agent_id": sc["agent_id"], "spawn_call_id": e["call_id"],
                                     "n_lines": sc["n_lines"], "tool_mix": sc["tool_mix"], "path": sc["path"],
                                     "first_ts": sc["first_ts"], "last_ts": sc["last_ts"]})
    if side_atif:
        flags.append(f"atif_sidechain_steps:{len(side_atif)}")

    persisted = sorted(os.path.relpath(p, trial_dir) for p in glob.glob(os.path.join(sess_dir, "tool-results", "*")))
    return {
        "ir_version": IR_VERSION, "trial": trial, "task": task, "reward": reward,
        "session_id": sid, "project_slug": slug, "cc_version": cc_version,
        "model": next((s["model"] for s in out_steps if s.get("model") and s["model"] != "<synthetic>"), None),
        "original_agent_config": {k: cfg.get("agent", {}).get(k) for k in ("name", "model_name", "kwargs")},
        "original_env_type": (cfg.get("environment") or {}).get("type"),
        "agent_timeout_s": timeout, "start_ts": steps[0]["timestamp"] if steps else None,
        "sources": {"session_jsonl": os.path.relpath(sess, trial_dir), "session_sha256": _sha256(sess),
                    "atif": "agent/trajectory.json" if atif else None, "atif_aligned": aligned,
                    "n_session_lines": len(recs), "persisted_outputs": persisted},
        "flags": flags, "sidechains": sidechains, "steps": out_steps,
    }


def _attachment_text(a):
    t = a.get("type")
    if t == "task_reminder":
        return json.dumps(a.get("content"), ensure_ascii=False)
    if t in ("queued_command",):
        p = a.get("prompt")
        return p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
    if t == "edited_text_file":
        return f"{a.get('filename')}: {a.get('snippet', '')}"
    if t in ("skill_listing",):
        return a.get("content", "")
    if t == "agent_listing_delta":
        return "\n".join(a.get("addedLines") or [])
    if t == "date_change":
        return json.dumps({k: v for k, v in a.items() if k != "type"})
    return json.dumps({k: v for k, v in a.items() if k != "type"}, ensure_ascii=False)
