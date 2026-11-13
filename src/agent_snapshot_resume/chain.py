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


# ------------------------------------------------------------ hint materialisation
def _hint_block(hint: str) -> dict[str, Any]:
    from .proxy.resume_proxy import hint_block_text
    return {"type": "text", "text": hint_block_text(hint)}


def hint_block(hint: str) -> str:
    """The exact text block the proxy injects for a hint (also what the session seed must carry)."""
    from .proxy.resume_proxy import hint_block_text
    return hint_block_text(hint)


def _content(rec: dict[str, Any]) -> list[dict[str, Any]]:
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, list):
        return c
    return [{"type": "text", "text": c}] if isinstance(c, str) else []


def _occurrences(ctx: list[dict[str, Any]], hint: str) -> list[tuple[int, int]]:
    want = _hint_block(hint)["text"]
    return [(i, j) for i, r in enumerate(ctx) for j, b in enumerate(_content(r))
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text") == want]


def _find_tool_result(records: list[dict[str, Any]], tool_use_id: str, cut: int) -> int | None:
    for i in range(min(cut, len(records))):
        for b in _content(records[i]):
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tool_use_id:
                return i
    return None


def _summary_index(records: list[dict[str, Any]], cut: int) -> int | None:
    for i in range(min(cut, len(records))):
        r = records[i]
        if r.get("type") == "user" and r.get("isCompactSummary"):
            return i
    return None


def materialize_hints(records: list[dict[str, Any]], hints: list[str], anchors: list[dict[str, Any]],
                      cut: int, *, persistent: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Put each hint exactly once into the model context of ``records[:cut]``.

    The hint is appended to the record it was injected on (the message carrying the anchor tool result). When that
    record is no longer in the model context (a compaction happened after the injection) the hint is appended to
    the compaction summary instead — but only if the previous run actually re-injected it after its compactions
    (``persistent``, i.e. its spec had ``persist_hint``); otherwise the hint was *gone* from that run's requests and
    seeding it would invent an instruction the model never saw.
    """
    out = list(records)
    audit: list[dict[str, Any]] = []
    for pos, hint in enumerate(hints):
        ctx = context_lines(out[:cut])
        occ = _occurrences(ctx, hint)
        if len(occ) > 1:
            raise ValueError(f"hint {pos} is in the model context {len(occ)} times; refusing to seed a duplicate")
        if len(occ) == 1:
            audit.append({"hint": pos, "where": "already_present", "context_message": occ[0][0]})
            continue
        anchor = anchors[pos] if pos < len(anchors) else {}
        tid = anchor.get("tool_use_id")
        idx = _find_tool_result(out, tid, cut) if tid else None
        ctx_ids = {id(r) for r in ctx}
        if idx is None or id(out[idx]) not in ctx_ids:
            if not persistent:
                raise ValueError(f"hint {pos} is no longer in the model context (compacted away) and the previous run "
                                 "did not persist hints: it cannot be re-seeded without inventing a turn the model "
                                 "never saw")
            sidx = _summary_index(out, cut)
            if sidx is None:
                raise ValueError(f"hint {pos}: anchor tool result {tid} is not in the model context and there is no "
                                 "compaction summary to carry the hint")
            idx = sidx
            where = "compaction_summary"
        else:
            where = "anchor_tool_result"
        body = list(_content(out[idx]))
        body = body + [_hint_block(hint)]
        msg = dict(out[idx].get("message") or {})
        msg["content"] = body
        out[idx] = {**out[idx], "message": msg}
        audit.append({"hint": pos, "where": where, "line": idx, "tool_use_id": tid,
                      "block_index": len(body) - 1})
    ctx = context_lines(out[:cut])
    for pos, hint in enumerate(hints):
        if len(_occurrences(ctx, hint)) != 1:
            raise ValueError(f"hint {pos} is not in the model context exactly once after materialisation")
    ordered = []
    for hint in hints:
        (mi, bi), = _occurrences(ctx, hint)
        ordered.append((mi, bi))
    if ordered != sorted(ordered):
        raise ValueError("materialised hints are not in injection order (older hints must come first)")
    return out, audit


def anchors_for(records: list[dict[str, Any]], hints: list[str], cut: int) -> list[dict[str, Any]]:
    """`hint_anchors` spec entries: each hint block bound to the tool result it follows in the same message."""
    ctx = context_lines(records[:cut])
    out = []
    for hint in hints:
        (mi, bi), = _occurrences(ctx, hint)
        ids = [b.get("tool_use_id") for b in _content(ctx[mi])[:bi]
               if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")]
        if ids:
            out.append({"block": _hint_block(hint)["text"], "tool_use_id": ids[-1]})
    return out


# ------------------------------------------------------------------- budget
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


def check_base_snapshot(record: dict[str, Any], *, trial: str, session_id: str | None,
                        raw_lines: list[str]) -> None:
    """Bind a mid-run snapshot record to the run it claims to come from, before it is used as an environment.

    The record must prove (a) it was taken from a clone of *this* trial's session, (b) its freeze point was
    measurable (a suspension the guest's ticker saw, no tool call in flight) and (c) the session prefix in the image
    is byte-identical to the first `session_line` lines of this run's session. Without (c) a record from another run
    (or another cut) would silently transplant a foreign filesystem into this resume.
    """
    import hashlib

    need = ("snapshot_id", "session_line", "prefix_sha256")
    missing = [k for k in need if not record.get(k)]
    if missing:
        raise ValueError(f"base snapshot record lacks {missing} (taken by an older snapshotter?): re-snapshot it")
    if record.get("trial") and record["trial"] != trial:
        raise ValueError(f"base snapshot belongs to trial {record['trial']}, not {trial}")
    if record.get("session_id") and session_id and record["session_id"] != session_id:
        raise ValueError("base snapshot was taken from a different session than this run's")
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
