"""Request parity: the resumed agent's first model request vs the ORIGINAL run's request at the same point.

The original run's passthrough proxy logged one record per /v1/messages request (`requests-*.jsonl`: model,
n_messages, max_tokens, thinking, n_tools, req_bytes, usage, ...). The request that produced resume step k is located
by step k's token usage (input/cache-read/cache-creation/output tokens are persisted with the assistant message in
the session and logged by the proxy for the response). Counting assistant messages is not enough: failed upstream
attempts and retries are logged too, and every context compaction adds one request.

The resumed first request is normalised the way the original route transformed requests before logging them
(`--drop-fields`, `--strip-is-error` of the passthrough proxy; the proxy re-serialises with `json.dumps` only when one
of those is active, else it forwards Claude Code's compact JSON), and the injected hint block(s) are removed. Then
the byte length must equal the original `req_bytes` exactly, and model/max_tokens/thinking/n_tools/n_messages must
be equal. The same check runs inside the resume proxy as a fail-closed gate (spec key `first_request_parity`).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

from .proxy.resume_proxy import (block_digest, compare_parity, route_from_passthrough_args, system_digest,
                                 tool_digest)

FIELDS = ("model", "n_messages", "max_tokens", "thinking", "n_tools")


def original_requests(trial_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """/v1/messages records of the original run's passthrough proxy, in log order."""
    out: list[dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(trial_dir, "agent", "model-proxy", "requests-*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("path") == "/v1/messages" and rec.get("req_bytes") is not None:
                        out.append(rec)
    return out


def _usage_key(u: dict[str, Any] | None):
    if not u:
        return None
    return (u.get("input_tokens"), u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0,
            u.get("output_tokens"))


def locate_request(ir: dict[str, Any], k_index: int, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the original request that produced step k. Returns {index, method, record} or {error}.

    method "usage": the unique logged request whose response usage equals step k's usage.
    method "count": fallback when usage is missing/ambiguous: k-th successful main-agent request, counting
    distinct main-chain assistant message ids and compactions before step k (only valid without failed attempts).
    """
    st = ir["steps"][k_index]
    key = _usage_key(st.get("usage"))
    if key and None not in key[:1]:
        hits = [i for i, r in enumerate(requests) if _usage_key(r.get("usage")) == key]
        if len(hits) == 1:
            return {"index": hits[0], "method": "usage", "record": requests[hits[0]]}
        if len(hits) > 1:
            return {"error": f"step usage matches {len(hits)} original requests (ambiguous)", "candidates": hits}
    ok = [i for i, r in enumerate(requests) if r.get("status") == 200 and r.get("stop_reason")]
    n_ids = len({s.get("message_id") for s in ir["steps"][:k_index] if s["source"] == "agent" and s.get("message_id")})
    n_compact = sum(1 for s in ir["steps"][:k_index] for e in s["events"] if e["type"] == "compaction_boundary")
    n = n_ids + n_compact
    if n < len(ok) and len(ok) == len(requests):
        return {"index": ok[n], "method": "count", "record": requests[ok[n]]}
    return {"error": "original request for step k not identifiable (no unique usage match, failed attempts logged)"}


def content_reference(session_lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Model-visible content the first resumed request must carry, from the truncated session prefix:
    the system prompt and tools of the agent's last persisted `prompt_snapshot` attachments (Claude Code 2.1.280
    writes them; absent -> not checked), and the digest of every message block of the context rebuilt after the last
    compaction (equiv.context_lines), in order."""
    from .equiv import blocks_from_session, context_lines
    cli = sysp = tools = None
    for d in session_lines:
        a = d.get("attachment") or {}
        if a.get("type") == "prompt_snapshot":
            cli = a.get("cliPrefix", cli)
            sysp = a.get("systemPrompt", sysp)
            tools = a.get("tools", tools)
    ctx = context_lines(session_lines)
    ref: dict[str, Any] = {"message_block_sha256": [block_digest(b) for b in blocks_from_session(ctx)],
                           "n_attachments": sum(1 for d in ctx if d.get("type") == "attachment")}
    if cli is not None and sysp is not None:
        ref["system_sha256"] = system_digest([cli, "\n\n".join(sysp)])
    if tools:
        ref["tools"] = {t["name"]: tool_digest({"name": t["name"], "description": t.get("description"),
                                                 "input_schema": (t.get("schema") or {}).get("input_schema")})
                        for t in tools}
        # Claude Code adds this deferred-loading placeholder to the request but not to the snapshot
        ref["extra_tools_allowed"] = ["DeferredToolPlaceholder"]
    return ref


def expected_from_record(rec: dict[str, Any], *, index: int, method: str) -> dict[str, Any]:
    return {"index": index, "method": method, "req_bytes": rec["req_bytes"],
            **{f: rec.get(f) for f in FIELDS}}


def check_files(bundle_dir: str | os.PathLike, request_path: str | os.PathLike,
                passthrough_args: list[str] | None = None) -> dict[str, Any]:
    """Offline parity of a captured first_request_rewritten.json against the bundle's recorded original request."""
    b = Path(bundle_dir)
    meta = json.loads((b / "bundle.json").read_text())
    spec = json.loads((b / "proxy_spec.json").read_text())
    exp = meta.get("orig_request_at_k")
    if not exp:
        return {"ok": False, "error": "bundle has no orig_request_at_k (original proxy log missing?)"}
    rq = json.loads(Path(request_path).read_text())
    body = rq.get("body", rq)
    hints = list(spec.get("hints") or ([spec["hint"]] if spec.get("hint") else []))
    rep = compare_parity(body, exp, route_from_passthrough_args(passthrough_args), hints)
    if exp.get("content"):
        from .proxy.resume_proxy import check_content
        rep["content_problems"] = check_content(body, exp["content"], hints)
        rep["ok"] = rep["ok"] and not rep["content_problems"]
    rep.update(bundle=meta["run_id"], orig_index=exp.get("index"), locate_method=exp.get("method"))
    return rep
