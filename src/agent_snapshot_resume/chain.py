"""Chained round-N resumes: seed a resumed run from the previous resumed run's own session.

A resumed run (round N-1) records everything about itself in its trial dir:

    agent/resume/proxy_spec.json    mode, sentinel, hint, prior_hints, persist_hint, resume_artifacts
    agent/resume/bundle.json        the parent bundle (agent_budget_s, cut_line, model, ...)
    agent/resume/run_start.json     when this run started (ephemeral budget accounting)
    agent/resume-proxy/<run>/first_request_rewritten.json   the exact first request (hint blocks + anchors)
    agent/sessions/projects/*/*.jsonl                       this run's own session (prefix copy + new turns)
    agent/model-proxy/requests-*.jsonl                      request times (ephemeral budget at the new cut)

Its session file starts with a copy of the previous bundle's ``session_trunc``, then the resume scaffolding
Claude Code wrote when it started (continue turn, synthetic assistant reply, sentinel prompt, resume
attachments), then the run's own turns. Hint blocks are injected on the wire and never persisted, so a chained
bundle must

  1. strip the previous run's scaffolding (it is not content the next resume should inherit),
  2. materialise every hint the run carried at the position the proxy put it (anchored on the tool result it
     followed, per the run's own first request), re-applying it to the compaction summary when the new cut is
     after a compaction,
  3. bind the result: the exact hint blocks, their anchors and the forbidden (old) sentinels go into
     ``proxy_spec.json`` so the proxy refuses any request that lost, duplicated or moved a hint.

Everything is fail-closed: an unexpected session shape raises instead of producing a seed whose history differs
from what the model saw.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from .equiv import context_lines

CONTINUE_TXT = "Continue from where you left off."
SYNTH_TXT = "No response requested."
NON_CONTENT_TYPES = {"queue-operation", "last-prompt", "mode", "atis-latch", "summary", "system"}


# ------------------------------------------------------------------ evidence
def run_evidence(trial_dir: Path) -> dict[str, Any]:
    """What a resumed run recorded about itself. Empty dict for a first-round (L0) trial."""
    trial_dir = Path(trial_dir)
    resume = trial_dir / "agent" / "resume"
    spec_path = resume / "proxy_spec.json"
    if not spec_path.is_file():
        return {}
    spec = json.loads(spec_path.read_text())
    parent = json.loads((resume / "bundle.json").read_text()) if (resume / "bundle.json").is_file() else {}
    start = json.loads((resume / "run_start.json").read_text()) if (resume / "run_start.json").is_file() else {}
    first = None
    paths = sorted(glob.glob(str(trial_dir / "agent" / "resume-proxy" / "*" / "first_request_rewritten.json")))
    if paths:
        first = json.loads(Path(paths[-1]).read_text())
    sessions = sorted(glob.glob(str(trial_dir / "agent" / "sessions" / "projects" / "*" / "*.jsonl")))
    return {"spec": spec, "parent_bundle": parent, "run_start": start, "first_request": first,
            "first_request_path": paths[-1] if paths else None,
            "session_paths": sessions, "resume_dir": str(resume)}


def hints_of(spec: dict[str, Any]) -> list[str]:
    """The hints a run carried, in injection order (oldest first)."""
    out = [h for h in (spec.get("prior_hints") or []) if h]
    if spec.get("hint"):
        out.append(spec["hint"])
    return out


def first_request_hints(body: dict[str, Any], hints: list[str]) -> list[dict[str, Any]]:
    """Where each hint block sits in the run's first request: the message it was appended to and the tool result
    it followed (the proxy appends the block to the last user message, keeping its tool results)."""
    from .proxy.resume_proxy import hint_block_text

    blocks: list[tuple[int, int, dict[str, Any]]] = []
    for mi, m in enumerate(body.get("messages") or []):
        c = m.get("content")
        for bi, b in enumerate(c if isinstance(c, list) else [{"type": "text", "text": c}]):
            if isinstance(b, dict):
                blocks.append((mi, bi, b))
    out = []
    for hint in hints:
        want = hint_block_text(hint)
        found = [(mi, bi) for mi, bi, b in blocks if b.get("type") == "text" and b.get("text") == want]
        if len(found) != 1:
            raise ValueError(f"the run's first request carries the hint block {len(found)} times, expected 1")
        mi, bi = found[0]
        ids = [b.get("tool_use_id") for m2, b2, b in blocks if m2 == mi and b2 < bi
               and b.get("type") == "tool_result" and b.get("tool_use_id")]
        if not ids:
            # The proxy only starts a new message when it cannot append to the anchor; the anchor rule below would
            # then be wrong, so refuse instead of guessing where the hint belongs.
            raise ValueError("the run's first request has no tool result before a hint block (new-message injection "
                             "is not supported by chained builds)")
        out.append({"hint": hint, "block": want, "msg_index": mi, "block_index": bi, "tool_use_id": ids[-1]})
    return out


# ------------------------------------------------------------- scaffolding strip
def _texts(rec: dict[str, Any]) -> list[str]:
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, str):
        return [c]
    if isinstance(c, list):
        return [b.get("text") or "" for b in c if isinstance(b, dict) and b.get("type") == "text"]
    return []


def _is_user_text(rec: dict[str, Any], txt: str) -> bool:
    return rec.get("type") == "user" and any(t.strip() == txt for t in _texts(rec))


def _is_synthetic_assistant(rec: dict[str, Any], txt: str) -> bool:
    return rec.get("type") == "assistant" and any(t.strip() == txt for t in _texts(rec))


def strip_scaffolding(records: list[dict[str, Any]], sentinel: str, continue_txt: str = CONTINUE_TXT,
                      synth_txt: str = SYNTH_TXT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop the previous run's resume scaffolding and reconnect the parent chain across the hole.

    The range removed is exactly what the proxy dropped from that run's requests: the continue turn, the synthetic
    assistant reply, the sentinel prompt and every non-assistant message after it (resume-time environment/date/
    token attachments). A continue block that Claude Code merged into the anchor message is removed from that
    message instead.
    """
    out = list(records)
    audit: list[dict[str, Any]] = []
    while True:
        idx = next((i for i, r in enumerate(out) if _is_user_text(r, sentinel)), None)
        if idx is None:
            break
        lo = idx
        if lo - 1 >= 0 and _is_synthetic_assistant(out[lo - 1], synth_txt):
            lo -= 1
        if lo - 1 >= 0 and _is_user_text(out[lo - 1], continue_txt):
            lo -= 1
        hi = idx + 1
        while hi < len(out) and out[hi].get("type") != "assistant":
            hi += 1
        removed = out[lo:hi]
        audit.append({"removed_lines": [i for i in range(lo, hi)], "reason": "resume scaffolding",
                      "uuids": [r.get("uuid") for r in removed if r.get("uuid")],
                      "types": [r.get("type") for r in removed]})
        anchor = out[lo - 1].get("uuid") if lo - 1 >= 0 else None
        dead = {r.get("uuid") for r in removed if r.get("uuid")}
        out = out[:lo] + out[hi:]
        if anchor and lo < len(out) and out[lo].get("parentUuid") in dead:
            out[lo] = {**out[lo], "parentUuid": anchor}
            audit[-1]["reconnected"] = {"line": lo, "to": anchor}
    if not audit:
        # A run whose session has no sentinel line cannot be chained safely: the next resume would inherit a
        # history whose last model-visible turn is the previous prompt, not the state the proxy injected into.
        raise ValueError("no resume sentinel found in the session: not a resumed run (or the sentinel was edited)")
    # Claude Code can keep the continue cue as a block of the last user message instead of its own turn.
    for i, r in enumerate(out):
        if r.get("type") != "user" or not isinstance((r.get("message") or {}).get("content"), list):
            continue
        body = list(r["message"]["content"])
        keep = [b for b in body if not (isinstance(b, dict) and b.get("type") == "text"
                                        and (b.get("text") or "").strip() == continue_txt)]
        if len(keep) != len(body):
            out[i] = {**r, "message": {**r["message"], "content": keep}}
            audit.append({"reason": "continue block inside the anchor message", "line": i})
    left = [i for i, r in enumerate(out)
            if _is_user_text(r, sentinel) or _is_user_text(r, continue_txt) or _is_synthetic_assistant(r, synth_txt)]
    if left:
        raise ValueError(f"resume scaffolding remains at lines {left[:5]}")
    return out, audit


# ------------------------------------------------------------ carried hints (where the parent's proxy had them)
def hint_block(hint: str) -> str:
    """The exact text block the proxy injects for a hint (also what a request must carry)."""
    from .proxy.resume_proxy import hint_block_text
    return hint_block_text(hint)


def _text_of_record(rec: dict[str, Any]) -> str:
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    return "".join(b.get("text") or "" for b in c if isinstance(b, dict) and b.get("type") == "text")


def _first_request_summary_text(body: dict[str, Any]) -> str | None:
    """The compact summary the run's own first request carried, if it was already resumed from a compaction."""
    from .proxy.resume_proxy import COMPACT_SUMMARY_PREFIX
    for m in body.get("messages") or []:
        c = m.get("content")
        texts = [c] if isinstance(c, str) else [b.get("text") or "" for b in (c or []) if isinstance(b, dict)]
        if any(COMPACT_SUMMARY_PREFIX in t for t in texts):
            return next(t for t in texts if COMPACT_SUMMARY_PREFIX in t)
    return None


def _summary_index(records: list[dict[str, Any]], cut: int) -> int | None:
    """Index of the compaction summary that is *in the model context* of records[:cut].

    A resumed run that compacted several times has one summary per boundary, and only the last one is part of the
    context Claude Code rebuilds on --resume. Appending the carried hint to an earlier summary would put it into a
    session line the model never sees, so the seeded history would silently lose the hint."""
    ctx_ids = {id(r) for r in context_lines(records[:cut])}
    for i in range(min(cut, len(records)) - 1, -1, -1):
        r = records[i]
        if r.get("type") == "user" and r.get("isCompactSummary") and id(r) in ctx_ids:
            return i
    return None


def carried_hint_positions(evidence: dict[str, Any], stripped: list[dict[str, Any]], cut: int,
                           carried: list[str]) -> list[dict[str, Any]]:
    """Where the previous run's proxy held each hint it carried, at this cut.

    A resumed run has no copy of its hints in its session: Claude Code rebuilds every request from the session and
    the proxy re-inj*ects* them each time (``inject_hint`` while the sentinel region is in the request,
    ``reinject_hint_after_compaction`` on the summary after a compaction). So a chained seed cannot carry them in the
    session either (a text block riding a tool_result record, a separate user turn and a modified compact summary are
    all dropped by ``--resume``); the chained run re-injects them at the same place instead:

    * pre-compaction: the message carrying the anchor tool result of the parent's own injection (the proxy appends
      the block to that message, and the hint stays there while the message stays in the context);
    * post-compaction: the compaction summary that is in this cut's context (``persist_hint``).

    ``first_request_hints`` refuses when the parent's first request has no tool result before the hint, so a hint that
    was injected as a new message cannot be placed and the build fails closed.
    """
    anchors = first_request_hints(evidence["first_request"]["body"], carried)
    # Is the summary the run sees at this cut the SAME summary it had when its proxy injected the hint? Then the
    # injection message (a tool-result turn) is still the place the hint lives; a LATER compaction moved it onto
    # the new summary (the parent's reinject_after_compaction). Comparing the texts decides which regime applies.
    sidx = _summary_index(stripped, cut)
    cut_summary = _text_of_record(stripped[sidx]) if sidx is not None else None
    parent_summary = _first_request_summary_text(evidence["first_request"]["body"])
    on_summary = cut_summary is not None and cut_summary != parent_summary
    if on_summary and not (evidence.get("spec") or {}).get("persist_hint"):
        # Its own proxy dropped the hint at the first compaction, so every later request of that run carried no
        # hint: re-injecting one here would give the model an instruction it never saw.
        raise ValueError("the previous run's hint was compacted away and it did not persist hints (persist_hint off), "
                         "so its own requests after that compaction carried no hint; this cut cannot be made faithful")
    out: list[dict[str, Any]] = []
    for a in anchors:
        entry = {"hint": a["hint"], "block": a["block"], "tool_use_id": a["tool_use_id"]}
        if on_summary:
            entry["on_summary"] = True
        out.append(entry)
    return out


def ts_seconds(ts: str) -> float:
    """Epoch seconds of a session/request timestamp (CC writes UTC ISO8601 with milliseconds)."""
    import datetime as dt
    s = ts.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(s).timestamp()


def cut_after_strip(original: list[dict[str, Any]], stripped: list[dict[str, Any]], cut: int) -> int:
    """Line index of the same step in the stripped session (record order is preserved, uuids identify records)."""
    where = {r.get("uuid"): i for i, r in enumerate(stripped) if r.get("uuid")}
    for rec in original[cut:]:
        u = rec.get("uuid")
        if u and u in where:
            return where[u]
    return len(stripped)


def tail_actions(actions: list[dict[str, Any]], session_line: int) -> list[dict[str, Any]]:
    """Actions a snapshot at `session_line` does not already contain: those observed at or after it.

    `session_line` is a *count* (the image holds lines 0..session_line-1) while `obs_line` is a 0-based index, so an
    observation at index == session_line is not in the image and must be replayed. An action's effect is complete
    once its observation is written, so every action observed before that line is in the snapshot's filesystem.
    Actions without a line (sub-agent calls) are replayed.
    """
    out = []
    for a in actions:
        line = a.get("obs_line")
        if line is None or int(line) >= int(session_line):
            out.append(a)
    return out


def check_base_snapshot(record: dict[str, Any], *, trial: str, raw_lines: list[str]) -> None:
    """Bind a mid-run snapshot record to the run it claims to come from, before it is used as an environment.

    The record must prove (a) it was taken from a clone of *this* trial, (b) its freeze point was measurable (a
    suspension the guest's ticker saw, no tool call in flight) and (c) the session prefix in the image is
    byte-identical to the first `session_line` lines of this run's session. (c) is what actually binds the history:
    a record from another run, session or cut cannot satisfy it. The receipt's `session_id` is the Harbor sandbox
    session (`<trial>__env`), not the agent's session id, so it is not used here.
    """
    import hashlib

    need = ("snapshot_id", "session_line", "prefix_sha256")
    missing = [k for k in need if not record.get(k)]
    if missing:
        raise ValueError(f"base snapshot record lacks {missing} (taken by an older snapshotter?): re-snapshot it")
    if record.get("trial") and record["trial"] != trial:
        raise ValueError(f"base snapshot belongs to trial {record['trial']}, not {trial}")
    if record.get("session_line_reliable") is False:
        raise ValueError("base snapshot's freeze point was not measurable (no tick gap or a tool call in flight); "
                         "it cannot be used for a tail replay")
    line = int(record["session_line"])
    if line > len(raw_lines):
        raise ValueError(f"base snapshot line {line} is past the end of this session ({len(raw_lines)} lines)")
    data = ("\n".join(raw_lines[:line]) + "\n").encode()
    if hashlib.sha256(data).hexdigest() != record["prefix_sha256"]:
        raise ValueError("base snapshot's session prefix does not match this run's session (wrong record, or the "
                         "session changed after the snapshot)")


# ------------------------------------------------------------------- budget
def remaining_budget(evidence: dict[str, Any], elapsed_s: float | None, fallback: float | None) -> tuple[float, str]:
    """Budget left in this run's own session: the parent budget minus time since this run started.

    ``ir.start_ts`` is the first line's timestamp, which a resumed session copies from the original run, so
    ``timeout - elapsed`` would charge the whole pre-resume history a second time.
    """
    budget = (evidence.get("parent_bundle") or {}).get("agent_budget_s")
    start = (evidence.get("run_start") or {}).get("t")
    if budget and start and elapsed_s is not None:
        left = float(budget) - float(elapsed_s)
        if left <= 0:
            raise ValueError(f"no budget left in the parent run ({budget} s - {elapsed_s} s elapsed)")
        return round(left, 1), "parent_budget_since_run_start"
    if fallback is None:
        raise ValueError("cannot derive the remaining budget: the parent bundle has no agent_budget_s / run_start")
    return round(float(fallback), 1), "timeout_minus_session_elapsed"
