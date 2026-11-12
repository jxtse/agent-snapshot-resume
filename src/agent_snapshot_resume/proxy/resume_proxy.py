"""Resume request-rewriting proxy (runs inside the sandbox on 127.0.0.1:4031, aiohttp only).

The agent's base URL is http://127.0.0.1:4031/r/<run_id>; it then POSTs /r/<run_id>/v1/messages.
Per-run spec: <spec_dir>/<run_id>.json
  {"mode": "same" | "hint",
   "sentinel": "...",                   # unique prompt the agent was resumed with
   "hint": "..." | null,                # mode=hint only
   "upstream": "http://127.0.0.1:4013" | "fake",   # fake = canned end_turn reply (dry run, no model call)
   "model": null | "model-id",          # optional body.model override
   "pin_date": "YYYY-MM-DD" | null,     # original run date
   "strip_headers": ["anthropic-beta-substring", ...],
   "resume_artifacts": {"continue_text": "...", "synthetic_assistant_text": "..."},   # adapter-specific; the
                                         # defaults are what Claude Code 2.1.207 `--resume --print` injects
   "require_sentinel", "require_text_only", "expected_first_request_sha256", "required_hint_blocks",
   "hint_anchors", "forbidden_sentinels": optional gates (see rewrite() / check_chained_history())}
Rewrites EVERY /v1/messages body (the agent re-sends its persisted resume artifacts on each turn):
  1. find the user message containing a text block with the sentinel -> drop it
  2. drop the preceding synthetic assistant message (e.g. 'No response requested.')
  3. drop notes the agent appended after the sentinel turn at resume time (e.g. '<file> was modified')
  4. drop the exact 'Continue from where you left off.' text block from the user message before it
  5. mode=hint: append ONE text block '<system-reminder>\n<hint>\n</system-reminder>' to that user message
  6. pin_date: rewrite "Today's date is YYYY-MM-DD." in the first user message to the original run date
  7. mode=hint + persist_hint: once the agent's own context compaction has removed the sentinel turn, append the
     same hint block ONCE to the (latest) compact-summary user message, so every later main-agent request still
     carries the hint exactly once (requests without sentinel and without summary, e.g. sub-agents, are untouched)
The proxy never handles credentials: it forwards the agent's (dummy) auth header to the passthrough proxy, which
replaces it. Logs: <log_dir>/<run_id>/requests.jsonl, first_request_{raw,rewritten}.json (auth headers removed).

    python resume_proxy.py --port 4031 --spec-dir /opt/asr-proxy/resume/specs --log-dir /logs/agent/resume-proxy
"""
import argparse, asyncio, hashlib, json, os, re, time

import aiohttp
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-authenticate",
       "proxy-authorization", "te", "trailers", "upgrade", "accept-encoding"}
AUTH = {"x-api-key", "authorization"}
CONTINUE_TXT = "Continue from where you left off."
SYNTH_TXT = "No response requested."
DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.")


def _blocks(m):
    c = m.get("content")
    return c if isinstance(c, list) else [{"type": "text", "text": c}] if isinstance(c, str) else []


def _artifact_texts(spec):
    art = spec.get("resume_artifacts") or {}
    return art.get("continue_text", CONTINUE_TXT), art.get("synthetic_assistant_text", SYNTH_TXT)


def _is_sentinel_msg(m, sentinel):
    return m.get("role") == "user" and any(b.get("type") == "text" and sentinel in (b.get("text") or "") for b in _blocks(m))


def assert_text_only(value):
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url", "input_image"):
            raise ValueError("Text-only continuation rejected an image block; no content was deleted or forwarded")
        for child in value.values():
            assert_text_only(child)
    elif isinstance(value, list):
        for child in value:
            assert_text_only(child)


def request_digest(body):
    # Only CC's transport identity is excluded. Every model-visible field is bound.
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != 'metadata'},
                                    sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def check_chained_history(body, spec):
    messages = body.get('messages') or []
    blocks = [(mi, bi, b) for mi, m in enumerate(messages) for bi, b in enumerate(_blocks(m))]
    prior = (-1, -1)
    for hint in spec.get('required_hint_blocks', []):
        found = [(mi, bi) for mi, bi, b in blocks if b.get('type') == 'text' and b.get('text') == hint]
        if len(found) != 1 or found[0] <= prior:
            raise ValueError('Chained history requires each exact hint once, in historical order')
        prior = found[0]
    for anchor in spec.get('hint_anchors', []):
        found = [(mi, bi) for mi, bi, b in blocks if b.get('type') == 'text' and b.get('text') == anchor['block']]
        if len(found) != 1:
            raise ValueError('Chained hint anchor missing or duplicated')
        mi, bi = found[0]
        if not any(b.get('type') == 'tool_result' and b.get('tool_use_id') == anchor['tool_use_id']
                   for b in _blocks(messages[mi])[:bi]):
            raise ValueError('Chained hint is not after its original tool result')
    if spec.get('forbidden_sentinels'):
        serialized = json.dumps(messages, ensure_ascii=False)
        if any(s and s in serialized for s in spec['forbidden_sentinels']):
            raise ValueError('Resume sentinel remains in model-visible history')


COMPACT_SUMMARY_PREFIX = "This session is being continued from a previous conversation"


def hint_block_text(hint):
    return f"<system-reminder>\n{hint}\n</system-reminder>"


def _text_of(b):
    return (b.get("text") or "") if isinstance(b, dict) and b.get("type") == "text" else ""


def find_compact_summary(msgs):
    """Index of the latest user message carrying the agent's compaction summary, else None."""
    idx = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and any(COMPACT_SUMMARY_PREFIX in _text_of(b) for b in _blocks(m)):
            idx = i
    return idx


def hint_stats(body, hint):
    """(exact hint blocks, verbatim hint occurrences in all message text) of a request body."""
    if not hint:
        return 0, 0
    blk = hint_block_text(hint)
    texts = [_text_of(b) for m in body.get("messages") or [] for b in _blocks(m)]
    return sum(t == blk for t in texts), sum(t.count(hint) for t in texts)


def reinject_after_compaction(body, spec, acts):
    """Post-compaction persistence: the sentinel turn is gone (summarised away) -> put the hint on the summary."""
    msgs = body.get("messages") or []
    c = find_compact_summary(msgs)
    if c is None:
        acts.append("no_anchor")
        return body
    hint = spec["hint"]
    bl = [dict(b) for b in _blocks(msgs[c])]
    if any(hint in _text_of(b) for m in msgs for b in _blocks(m)):
        acts.append("hint_already_present")  # e.g. the summary quotes it verbatim: never duplicate
        return body
    blk = {"type": "text", "text": hint_block_text(hint)}
    # keep a cache breakpoint (if any) on the message's last block
    if bl and "cache_control" in bl[-1]:
        cc = bl[-1].pop("cache_control")
        blk["cache_control"] = cc
    new = list(msgs)
    new[c] = dict(msgs[c], content=bl + [blk])
    acts.append(f"reinject_hint_after_compaction@{c}")
    return dict(body, messages=new)


def rewrite(body, spec):
    """Return (new_body, actions). Pure function; unit-testable."""
    if spec.get("require_text_only"):
        assert_text_only(body)
    acts = []
    msgs = body.get("messages") or []
    sentinel = spec["sentinel"]
    continue_txt, synth_txt = _artifact_texts(spec)
    s = next((i for i, m in enumerate(msgs) if _is_sentinel_msg(m, sentinel)), None)
    if s is None:
        if spec.get('require_sentinel'):
            raise ValueError('Required resume sentinel missing; refusing unverified continuation')
        acts.append("sentinel_not_found")
        if spec.get("mode") == "hint" and spec.get("hint") and spec.get("persist_hint"):
            body = reinject_after_compaction(body, spec, acts)
        return body, acts
    dropped = []
    # extra (non-sentinel) blocks CC attached to the sentinel turn at resume time
    for b in _blocks(msgs[s]):
        if not (b.get("type") == "text" and sentinel in (b.get("text") or "")):
            dropped.append({"from": "sentinel_msg", "type": b.get("type"), "text": (b.get("text") or "")[:500]})
    lo = s
    if s >= 1 and msgs[s - 1].get("role") == "assistant" and all(
            b.get("type") == "text" and (b.get("text") or "").strip() == synth_txt for b in _blocks(msgs[s - 1])):
        lo = s - 1
        acts.append("drop_synthetic_assistant")
    # notes CC generates at resume time (e.g. "<file> was modified" because replayed Bash changed a file CC had
    # cached) land right after the sentinel turn; they were not in the original context -> drop and log them.
    e = s + 1
    while e < len(msgs) and msgs[e].get("role") != "assistant":
        for b in _blocks(msgs[e]):
            dropped.append({"from": "post_sentinel", "role": msgs[e].get("role"), "type": b.get("type"), "text": (b.get("text") or "")[:500]})
        e += 1
    if e > s + 1:
        acts.append(f"drop_post_sentinel_msgs:{e - s - 1}")
    new = msgs[:lo] + msgs[e:]
    acts.append(f"drop_sentinel_user@{s}")
    # Claude Code puts a prompt-cache breakpoint on the last block of the last message. At resume time that is a
    # dropped resume-time block (2.1.280: the post-sentinel environment/date note; 2.1.207: the sentinel turn), so
    # carry it to the last block of the new last message, as the original run's request at step k had it.
    removed_cc = [b["cache_control"] for m in msgs[lo:e] for b in _blocks(m) if isinstance(b, dict) and "cache_control" in b]
    tgt = lo - 1
    # CC can retain historical system reminders between tool results and resume scaffolding.
    # Keep those messages, but reach the adjacent user result to remove only the exact synthetic cue.
    while tgt >= 0 and new[tgt].get("role") == "system":
        tgt -= 1
    if tgt >= 0 and new[tgt].get("role") == "user":
        bl = _blocks(new[tgt])
        keep = [b for b in bl if not (b.get("type") == "text" and (b.get("text") or "").strip() == continue_txt)]
        if len(keep) != len(bl):
            acts.append("drop_continue_block")
        if spec.get("mode") == "hint" and spec.get("hint"):
            keep = keep + [{"type": "text", "text": hint_block_text(spec['hint'])}]
            acts.append("inject_hint")
        new[tgt] = dict(new[tgt], content=keep)
    elif spec.get("mode") == "hint" and spec.get("hint"):
        new.insert(lo, {"role": "user", "content": [{"type": "text", "text": hint_block_text(spec['hint'])}]})
        acts.append("inject_hint_new_msg")
    if removed_cc and new and not any("cache_control" in b for b in _blocks(new[-1]) if isinstance(b, dict)):
        last = new[-1]
        bl = [dict(b) for b in _blocks(last)]
        if bl:
            bl[-1]["cache_control"] = removed_cc[-1]
            new[-1] = dict(last, content=bl)
            acts.append("move_cache_breakpoint")
    if spec.get("pin_date") and new:
        bl = _blocks(new[0])
        for b in bl:
            if b.get("type") == "text" and DATE_RE.search(b.get("text") or ""):
                b["text"] = DATE_RE.sub(f"Today's date is {spec['pin_date']}.", b["text"], count=1)
                acts.append("pin_date")
                break
        new[0] = dict(new[0], content=bl)
    if spec.get("model"):
        body["model"] = spec["model"]
        acts.append("override_model")
    body = dict(body, messages=new)
    check_chained_history(body, spec)
    if dropped:
        acts.append({"dropped_blocks": dropped})
    return body, acts


def request_fingerprint(body, spec):
    """Per-request evidence fields: hint counts and a system/tools fingerprint (main agent vs sub-agent)."""
    sysv = body.get("system")
    sys_txt = sysv if isinstance(sysv, str) else "".join(_text_of(b) for b in (sysv or []))
    tools = sorted(t.get("name", "") for t in body.get("tools") or [] if isinstance(t, dict))
    blocks, occ = hint_stats(body, spec.get("hint") if spec.get("mode") == "hint" else None)
    return {"hint_blocks": blocks, "hint_occurrences": occ,
            "sys_sha": hashlib.sha256(sys_txt[:4000].encode()).hexdigest()[:12],
            "tools_sha": hashlib.sha256(json.dumps(tools).encode()).hexdigest()[:12], "n_tools": len(tools),
            "compact_summary": find_compact_summary(body.get("messages") or []) is not None}


def load_spec(app, run_id):
    p = os.path.join(app["spec_dir"], f"{run_id}.json")
    st = os.stat(p)
    cache = app["specs"].get(run_id)
    if cache and cache[0] == st.st_mtime:
        return cache[1]
    with open(p) as stream:
        spec = json.load(stream)
    if spec.get("key_file"):
        raise ValueError("key_file is not supported: credentials belong to the passthrough proxy only")
    app["counts"][run_id] = 0  # new/updated spec = new resume run: re-capture first request
    app["specs"][run_id] = (st.st_mtime, spec)
    return spec


def wlog(app, run_id, name, obj, mode="a"):
    d = os.path.join(app["log_dir"], run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), mode, encoding="utf-8") as f:
        if mode == "a":
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            json.dump(obj, f, ensure_ascii=False, indent=1)


def fake_reply(model, stream):
    if not stream:
        return web.json_response({"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                                  "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                                  "usage": {"input_tokens": 1, "output_tokens": 1}})
    evs = [("message_start", {"type": "message_start", "message": {"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
           ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
           ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}}),
           ("content_block_stop", {"type": "content_block_stop", "index": 0}),
           ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
           ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {e}\ndata: {json.dumps(p)}\n\n" for e, p in evs).encode()


async def handle(request):
    app = request.app
    run_id = request.match_info["run_id"]
    tail = request.match_info["tail"]
    try:
        spec = load_spec(app, run_id)
    except FileNotFoundError:
        return web.json_response({"type": "error", "error": {"type": "not_found", "message": f"no spec {run_id}"}}, status=404)
    except ValueError as exc:
        return web.json_response({"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}}, status=400)
    body = await request.read()
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "path": "/" + tail, "method": request.method}
    out_body = body
    is_msg = request.method == "POST" and tail.startswith("v1/messages") and "count_tokens" not in tail
    if is_msg and body:
        d = json.loads(body)
        n = app["counts"].get(run_id, 0)
        app["counts"][run_id] = n + 1
        if n == 0:
            hdr = {k: v for k, v in request.headers.items() if k.lower() not in AUTH}
            wlog(app, run_id, "first_request_raw.json", {"headers": hdr, "body": d}, mode="w")
        try:
            nd, acts = rewrite(d, spec)
            if spec.get('expected_first_request_sha256') and not spec.get('_first_request_validated'):
                if request_digest(nd) != spec['expected_first_request_sha256']:
                    raise ValueError('First model request differs from the frozen fake-resume request')
                spec['_first_request_validated'] = True
                acts.append('first_request_hash_gate_passed')
        except ValueError as exc:
            rec.update(status=400, local_gate_error=str(exc))
            wlog(app, run_id, "requests.jsonl", rec)
            return web.json_response({"type":"error", "error":{"type":"invalid_request_error", "message":str(exc)}}, status=400)
        if n == 0:
            wlog(app, run_id, "first_request_rewritten.json", {"body": nd, "actions": acts}, mode="w")
        out_body = json.dumps(nd, ensure_ascii=False).encode()
        rec.update({"n": n, "model": nd.get("model"), "n_messages_in": len(d.get("messages") or []),
                    "n_messages_out": len(nd.get("messages") or []), "actions": acts, "bytes": len(out_body)})
        rec.update(request_fingerprint(nd, spec))
        if any(isinstance(a, str) and a.startswith("reinject_hint_after_compaction") for a in acts):
            c = find_compact_summary(nd.get("messages") or [])
            key = hashlib.sha256(json.dumps(nd["messages"][c], ensure_ascii=False).encode()).hexdigest()[:12]
            if key not in app["compact_seen"].setdefault(run_id, set()):
                app["compact_seen"][run_id].add(key)
                wlog(app, run_id, f"post_compaction_request_{len(app['compact_seen'][run_id])}.json",
                     {"n": n, "body": nd, "actions": acts}, mode="w")
        if spec.get("upstream") == "fake":
            rec["status"] = 200
            wlog(app, run_id, "requests.jsonl", rec)
            if d.get("stream"):
                resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(fake_reply(nd.get("model"), True))
                await resp.write_eof()
                return resp
            return fake_reply(nd.get("model"), False)
    elif spec.get("upstream") == "fake":
        wlog(app, run_id, "requests.jsonl", dict(rec, status=404))
        return web.json_response({"type": "error", "error": {"type": "not_found", "message": "fake"}}, status=404)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    for h in spec.get("strip_headers") or []:
        if "anthropic-beta" in headers:
            headers["anthropic-beta"] = ",".join(x for x in headers["anthropic-beta"].split(",") if h not in x)
    t0 = time.time()
    url = spec["upstream"].rstrip("/") + "/" + tail + (("?" + request.query_string) if request.query_string else "")
    retries = 0
    try:
        while True:  # bounded retry of upstream rate limiting (request not processed upstream -> safe)
            up = await app["session"].request(request.method, url, data=out_body, headers=headers)
            if up.status in (429, 500, 502, 503, 504, 529) and retries < 400:
                await up.read()
                up.release()
                retries += 1
                await asyncio.sleep(min(60, 5 * retries) if retries > 8 else 30)
                continue
            break
        if retries:
            rec["upstream_retries"] = retries
    except Exception as e:  # noqa: BLE001
        rec.update({"status": 599, "error": repr(e)[:300]})
        wlog(app, run_id, "requests.jsonl", rec)
        return web.json_response({"type": "error", "error": {"type": "api_error", "message": "resume proxy upstream error"}}, status=529)
    resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in HOP | {"content-encoding"}})
    await resp.prepare(request)
    nbytes = 0
    async for chunk in up.content.iter_any():
        nbytes += len(chunk)
        await resp.write(chunk)
    await resp.write_eof()
    up.release()
    rec.update({"status": up.status, "latency_s": round(time.time() - t0, 2), "resp_bytes": nbytes})
    wlog(app, run_id, "requests.jsonl", rec)
    return resp


async def health(request):
    return web.json_response({"ok": True, "runs": len(request.app["counts"]), "counts": request.app["counts"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4031)
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args()
    app = web.Application(client_max_size=1 << 30)
    app.update(spec_dir=os.path.expanduser(a.spec_dir), log_dir=os.path.expanduser(a.log_dir), specs={}, counts={},
               compact_seen={})

    async def on_start(app):
        app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=3600), auto_decompress=False)

    async def on_stop(app):
        await app["session"].close()

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_get("/_health", health)
    app.router.add_route("*", "/r/{run_id}/{tail:.*}", handle)
    web.run_app(app, host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
