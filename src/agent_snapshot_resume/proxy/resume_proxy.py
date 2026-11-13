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
   "hint_anchors", "forbidden_sentinels": optional gates (see rewrite() / check_chained_history()),
   "max_model_requests": N,            # refuse model request N+1 and later (bounded continuation)
   "first_request_parity": {"req_bytes", "model", "n_messages", "max_tokens", "thinking", "n_tools", "route",
                            "content": {system_sha256, tools, extra_tools_allowed, message_block_sha256,
                                        n_attachments}}
                                         # fail-closed gate on the first request (see check_content); once it
                                         # failed, every later request of the run is refused too}
Rewrites EVERY /v1/messages body (the agent re-sends its persisted resume artifacts on each turn):
  1. find the user message containing a text block with the sentinel -> drop it
  2. drop the preceding synthetic assistant message (e.g. 'No response requested.')
  3. drop notes the agent appended after the sentinel turn at resume time (e.g. '<file> was modified')
  4. drop the exact 'Continue from where you left off.' text block from the user message before it
  5. mode=hint: append ONE text block '<system-reminder>\n<hint>\n</system-reminder>' to that user message
  6. pin_date: rewrite "Today's date is YYYY-MM-DD." in the first user message to the original run date (never in a
     compaction summary; the content keeps its shape: string or block list)
  7. mode=hint + persist_hint: once the agent's own context compaction has removed the sentinel turn, append the
     same hint block ONCE to the (latest) compact-summary user message, so every later main-agent request still
     carries the hint exactly once (requests without sentinel and without summary, e.g. sub-agents, are untouched)
The proxy never handles credentials: it forwards the agent's (dummy) auth header to the passthrough proxy, which
replaces it. Logs: <log_dir>/<run_id>/requests.jsonl, first_request_{raw,rewritten}.json (auth headers removed).

    python resume_proxy.py --port 4031 --spec-dir /opt/asr-proxy/resume/specs --log-dir /logs/agent/resume-proxy
"""
import argparse, asyncio, copy, hashlib, json, os, re, time

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


# ------------------------------------------------------------------ request parity (see agent_snapshot_resume.parity)
def route_from_passthrough_args(args):
    """The request transformations of the passthrough proxy flags (only the ones that change request bytes)."""
    args = list(args or [])
    route = {"drop_fields": [], "strip_is_error": False, "strip_thinking": False}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--drop-fields" and i + 1 < len(args):
            route["drop_fields"] = [x for x in args[i + 1].split(",") if x]
            i += 1
        elif a == "--strip-is-error":
            route["strip_is_error"] = True
        elif a == "--strip-thinking":
            route["strip_thinking"] = True
        i += 1
    return route


def remove_hint_blocks(body, hints):
    """Copy of body without the exact injected hint blocks. A breakpoint carried by a removed block goes back to the
    preceding block of the same message (where Claude Code had it before the injection)."""
    blocks = {hint_block_text(h) for h in hints if h}
    d = copy.deepcopy(body)
    n = 0
    if not blocks:
        return d, 0
    for m in d.get("messages") or []:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        keep = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text") in blocks:
                n += 1
                if "cache_control" in b and keep and isinstance(keep[-1], dict):
                    keep[-1] = dict(keep[-1], cache_control=b["cache_control"])
                continue
            keep.append(b)
        m["content"] = keep
    return d, n


def route_bytes(body, route):
    """Byte length of body as the original route's passthrough proxy logged it (req_bytes)."""
    d = copy.deepcopy(body)
    active = bool(route.get("drop_fields") or route.get("strip_is_error") or route.get("strip_thinking"))
    for k in route.get("drop_fields") or []:
        d.pop(k, None)
    for m in d.get("messages") or []:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        keep = []
        for b in c:
            if isinstance(b, dict) and route.get("strip_thinking") and b.get("type") in ("thinking", "redacted_thinking"):
                continue
            if isinstance(b, dict) and route.get("strip_is_error") and b.get("type") == "tool_result":
                b.pop("is_error", None)
            keep.append(b)
        m["content"] = keep
    if active:
        return len(json.dumps(d, ensure_ascii=False).encode())
    return len(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode())


def compare_parity(body, expected, route=None,
            hints=None):
    """Parity report of a (rewritten) first request body against the expected original request fields."""
    route = route if route is not None else expected.get("route") or {}
    stripped, n_hint = remove_hint_blocks(body, hints or [])
    got = {"req_bytes": route_bytes(stripped, route), "model": body.get("model"),
           "n_messages": len(body.get("messages") or []), "max_tokens": body.get("max_tokens"),
           "thinking": body.get("thinking"), "n_tools": len(body.get("tools") or [])}
    diffs = {k: {"got": got[k], "expected": expected.get(k)} for k in got if got[k] != expected.get(k)}
    return {"ok": not diffs, "diffs": diffs, "got": got, "expected": {k: expected.get(k) for k in got},
            "hint_blocks_removed": n_hint, "byte_diff": got["req_bytes"] - (expected.get("req_bytes") or 0)}


# ------------------------------------------------------------------ content parity (model-visible content)
# The request's model-visible content as comparable blocks: (role, type, ...) tuples, one per content block.
# Single source of truth for the proxy gate and for `asr equiv` (agent_snapshot_resume.equiv imports these).
REMINDER_SUFFIX_RE = re.compile(r"(?:\n\n<system-reminder>\n.*?\n</system-reminder>)+", re.S)


def norm_tr_content(c):
    if isinstance(c, str):
        return c
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def blocks_from_request(msgs):
    out = []
    for i, m in enumerate(msgs):
        role = m["role"]
        c = m["content"]
        if isinstance(c, str):
            out.append(((role, "text", c), i))
            continue
        for b in c:
            bt = b.get("type")
            if bt == "text":
                if role == "assistant" and not b.get("text"):
                    continue
                out.append(((role, "text", b["text"]), i))
            elif bt == "thinking":
                out.append(((role, "thinking", b.get("thinking", ""), b.get("signature")), i))
            elif bt == "redacted_thinking":
                out.append(((role, "redacted_thinking", b.get("data")), i))
            elif bt == "tool_use":
                out.append(((role, "tool_use", b["id"], b["name"],
                             json.dumps(b.get("input"), sort_keys=True, ensure_ascii=False)), i))
            elif bt == "tool_result":
                out.append(((role, "tool_result", b["tool_use_id"], bool(b.get("is_error")),
                             norm_tr_content(b.get("content"))), i))
            else:
                out.append(((role, bt, json.dumps({k: v for k, v in b.items() if k != "cache_control"},
                                                  sort_keys=True)), i))
    return out


def block_digest(blk):
    return hashlib.sha256(json.dumps(list(blk), ensure_ascii=False).encode()).hexdigest()


def strip_reminder_suffix(blk):
    """A tool_result whose string content ends with <system-reminder> blocks (Claude Code renders some attachments
    that way) -> the same block without them; None if there is no such suffix."""
    if blk[1] != "tool_result" or not isinstance(blk[4], str):
        return None
    m = re.search(r"(?:\n\n<system-reminder>\n.*?\n</system-reminder>)+$", blk[4], re.S)
    if not m or not m.group(0):
        return None
    return blk[:4] + (blk[4][:m.start()],)


ATTACHMENT_TEXT_PREFIXES = ("<system-reminder>", "<task-notification>")
_LINENO_RE = re.compile(r"(?m)^\s*\d+\t")  # Read-tool rendering "N<TAB>line"


def rendered_line_ok(line, attachments, templates):
    """A single line is a fixed rendering line, or a slot template whose every slot is filled by the named field
    (`<type>.<path>`) of one of `attachments`. attachments: [{"fields": {"<type>.<path>": [values]}, ...}]"""
    if line in (templates or {}).get("fixed", ()):
        return True
    for tpl, fields in (templates or {}).get("slots", ()):
        parts = tpl.split("\x00")
        if len(parts) - 1 != len(fields) or not (line.startswith(parts[0]) and line.endswith(parts[-1])):
            continue
        rest = line[len(parts[0]):len(line) - len(parts[-1]) if parts[-1] else len(line)]
        cands = [[v for a in attachments for v in a["fields"].get(f, ())] for f in fields]
        if _fill_slots(rest, parts[1:-1], cands):
            return True
    return False


def _fill_slots(rest, middles, cands):
    """rest = v1 m1 v2 ... vn with v_i in cands[i] (exact strings)."""
    if not middles:
        return rest in cands[0]
    for v in cands[0]:
        if rest.startswith(v + middles[0]) and _fill_slots(rest[len(v) + len(middles[0]):], middles[1:], cands[1:]):
            return True
    return False


def _leaf_forms(v):
    """How a persisted value appears as consecutive request lines: verbatim, or numbered as the Read tool renders
    file contents ("N<TAB>line"), with or without its final empty line."""
    lines = v.split("\n")
    forms = [lines] + ([lines[:-1]] if len(lines) > 1 and lines[-1] == "" else [])
    return forms + [[f"{k + 1}\t{x}" for k, x in enumerate(f)] for f in forms]


def rendered_text_problems(text, attachments, templates, consumed=None):
    """Parse rendered attachment text in order. Each line must start a whole persisted leaf value of one of
    `attachments` (multi-line values as one contiguous block, exactly), or be a fixed rendering line / filled slot
    template (rendered_line_ok). Reordered, duplicated, altered or foreign lines fail; the longest leaf match wins.
    A file header (slot template filled by exactly one `file` attachment) binds the following leaf values to that
    attachment until its content leaf (`content_leaf`: index of the file's content in its leaves) is matched; a
    file's header and its content come from one attachment, single-line contents included (no cross-file swaps). Each leaf is
    used at most once across all rendered blocks (`consumed`: shared set of (attachment id, leaf index)).
    attachments: [{"leaves": [...], "fields": {...}}]."""
    consumed = set() if consumed is None else consumed
    lines = text.split("\n")
    bad, i, bound = [], 0, None
    while i < len(lines):
        if lines[i] == "":
            i += 1
            continue
        best = None
        for a in attachments:
            if bound is not None and a is not bound:
                continue
            for li, v in enumerate(a.get("leaves", ())):
                key = (id(a), li)
                if key in consumed:
                    continue
                for f in _leaf_forms(v):
                    if f and lines[i:i + len(f)] == f and (best is None or len(f) > best[0]):
                        best = (len(f), key)
        if best:
            consumed.add(best[1])
            i += best[0]
            if bound is not None and best[1] == (id(bound), bound.get("content_leaf")):
                bound = None  # the bound file's content is done; later lines belong to any attachment
            continue
        owner = _template_owner(lines[i], attachments, templates)
        if owner is False:
            bad.append(lines[i])
        elif owner is not None or lines[i] not in (templates or {}).get("fixed", ()):
            bound = owner  # a file header binds; another slot template unbinds; fixed lines keep the binding
        i += 1
    return bad


def _template_owner(line, attachments, templates):
    """False: not a rendering line. Otherwise the single attachment whose fields fill a slot template (when the
    line is a file-style header naming exactly one attachment), else None."""
    if not rendered_line_ok(line, attachments, templates):
        return False
    owners = [a for a in attachments if rendered_line_ok(line, [a], {"slots": (templates or {}).get("slots", ())})]
    return owners[0] if len(owners) == 1 and owners[0].get("type") == "file" else None


def tool_digest(t):
    return hashlib.sha256(json.dumps([t.get("name"), t.get("description"), t.get("input_schema")],
                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def system_digest(texts):
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest()


def check_content(body, ref, hints):
    """Model-visible content of a rewritten request vs the reference built from the session prefix
    (agent_snapshot_resume.parity.content_reference). Returns a list of problems (empty = ok).

      system         sha256 of the system block texts == the prompt snapshot the agent persisted (if recorded)
      tools          every recorded tool present with the same name/description/input_schema; any other tool must
                     be listed in extra_tools_allowed
      messages       every reference block (session prefix after compaction rebuild) present in order, byte-identical
                     (a tool_result may carry an appended <system-reminder> suffix); every other block must be a
                     system-role message, an exact hint block, or user text that starts with a rendered-attachment
                     marker; their number must not exceed the prefix's attachment count, and every line of
                     them must be explained, in order, by the attachments persisted at the same context position:
                     whole leaf values as exact contiguous blocks (each used once), fixed rendering lines, or slot
                     templates filled field by field (rendered_text_problems)
      completeness   require_complete: fail if the prefix had no prompt snapshot (system/tools unbound)
    """
    problems = []
    if ref.get("system_sha256"):
        sysv = body.get("system")
        texts = [b.get("text") for b in sysv] if isinstance(sysv, list) else [sysv]
        if system_digest(texts) != ref["system_sha256"]:
            problems.append("system prompt differs from the agent's persisted prompt snapshot")
    if ref.get("tools"):
        got = {t.get("name"): tool_digest(t) for t in body.get("tools") or []}
        for name, dg in ref["tools"].items():
            if got.get(name) != dg:
                problems.append(f"tool {name} missing or different")
        allowed = set(ref.get("extra_tools_allowed") or [])
        extra = [n for n in got if n not in ref["tools"] and n not in allowed]
        if extra:
            problems.append(f"unexpected tools {extra}")
    if ref.get("require_complete") and not (ref.get("system_sha256") and ref.get("tools")):
        problems.append("content evidence incomplete: the session prefix has no prompt snapshot with the system "
                        "prompt and tools")
    want = ref.get("message_block_sha256")
    if want is not None:
        hint_blocks = {hint_block_text(h) for h in hints or []}
        j, extras, rendered = 0, 0, []
        for blk, _ in blocks_from_request(body.get("messages") or []):
            if j < len(want) and block_digest(blk) == want[j]:
                j += 1
                continue
            stripped = strip_reminder_suffix(blk)
            if j < len(want) and stripped is not None and block_digest(stripped) == want[j]:
                j += 1
                extras += 1
                rendered.append((j, blk[4][len(stripped[4]):]))
                continue
            if blk[1] == "text" and blk[2] in hint_blocks:
                continue
            if blk[0] == "system" or (blk[0] == "user" and blk[1] == "text"
                                      and blk[2].lstrip().startswith(ATTACHMENT_TEXT_PREFIXES)):
                extras += 1
                rendered.append((j, blk[2] if isinstance(blk[2], str) else json.dumps(blk[2], ensure_ascii=False)))
                continue
            problems.append(f"unexplained {blk[0]} {blk[1]} block before reference block {j}")
            break
        if j < len(want):
            problems.append(f"reference block {j} of {len(want)} missing or different")
        if extras > int(ref.get("n_attachments", 0)):
            problems.append(f"{extras} rendered-attachment blocks for {ref.get('n_attachments', 0)} attachments")
        # Every line of rendered attachment text must be explained by the attachments persisted at the same context
        # position (block index, +-1 for a reminder rendered onto the neighbouring tool result), field by field.
        if "attachments" in ref:
            templates = ref.get("rendered_templates")
            if templates is None:
                if rendered:
                    problems.append("rendered attachment text present but this agent version has no rendering "
                                    "templates: cannot bind it")
            else:
                bad, consumed = [], set()
                for pos, txt in rendered:
                    near = [a for a in ref["attachments"] if abs(a["pos"] - pos) <= 1]
                    bad += rendered_text_problems(txt, near, templates, consumed)
                if bad:
                    problems.append(f"rendered attachment text not explained by the attachments persisted at that "
                                    f"position: {[b[:120] for b in bad[:3]]}")
    return problems


def spec_hints(spec):
    """All hints a request may carry: the chain's earlier hints (spec "prior_hints", oldest first) + this run's."""
    hs = list(spec.get("prior_hints") or [])
    if spec.get("mode") == "hint" and spec.get("hint"):
        hs.append(spec["hint"])
    return hs


def pin_date_in_message(msg, pin_date, acts):
    """Rewrite the first "Today's date is ..." in msg to pin_date. Keeps the content's shape: a string stays a
    string (after a compaction the first message is the summary, sent as a plain string; turning it into a block
    list would change the request bytes). Messages without a date are returned unchanged."""
    repl = f"Today's date is {pin_date}."
    c = msg.get("content")
    if any(COMPACT_SUMMARY_PREFIX in _text_of(b) for b in _blocks(msg)):
        return msg  # never edit a compaction summary (its date reminder is a separate, persisted system message)
    if isinstance(c, str):
        if DATE_RE.search(c):
            acts.append("pin_date")
            return dict(msg, content=DATE_RE.sub(repl, c, count=1))
        return msg
    if not isinstance(c, list):
        return msg
    for i, b in enumerate(c):
        if isinstance(b, dict) and b.get("type") == "text" and DATE_RE.search(b.get("text") or ""):
            nb = dict(b, text=DATE_RE.sub(repl, b["text"], count=1))
            acts.append("pin_date")
            return dict(msg, content=c[:i] + [nb] + c[i + 1:])
    return msg


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
        new[0] = pin_date_in_message(new[0], spec["pin_date"], acts)
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
        cap = spec.get("max_model_requests")
        if cap and n >= int(cap):
            # bounded continuation (e.g. a parity probe): refuse further model calls; the agent ends with an error
            rec.update(n=n, status=400, local_gate_error=f"max_model_requests={cap} reached")
            wlog(app, run_id, "requests.jsonl", rec)
            return web.json_response({"type": "error", "error": {"type": "invalid_request_error",
                                      "message": f"resume proxy: max_model_requests={cap} reached"}}, status=400)
        if n == 0:
            hdr = {k: v for k, v in request.headers.items() if k.lower() not in AUTH}
            wlog(app, run_id, "first_request_raw.json", {"headers": hdr, "body": d}, mode="w")
        try:
            nd, acts = rewrite(d, spec)
            if spec.get("first_request_parity"):
                state = spec.get("_parity_state")  # lives on the cached spec: reset only by a new spec
                if state is False:
                    raise ValueError("first-request parity failed earlier in this run; every request is refused")
                if state is None:
                    # Fail closed unless the first resumed request equals the original run's request at the cut:
                    # bytes as logged by the original route, model/max_tokens/thinking/tool and message counts,
                    # and the model-visible content (system prompt, tools, every message block of the prefix).
                    exp = spec["first_request_parity"]
                    rep = compare_parity(nd, exp, exp.get("route") or {}, spec_hints(spec))
                    if exp.get("content"):
                        rep["content_problems"] = check_content(nd, exp["content"], spec_hints(spec))
                        rep["ok"] = rep["ok"] and not rep["content_problems"]
                    # provenance for --freeze-first-request: which exact request and route this report judged
                    rep.update(request_sha256=request_digest(nd), route=exp.get("route") or {},
                               upstream_fake=spec.get("upstream") == "fake")
                    wlog(app, run_id, "first_request_parity.json", rep, mode="w")
                    spec["_parity_state"] = bool(rep["ok"])
                    if not rep["ok"]:
                        raise ValueError("First resumed request differs from the original request at the cut: "
                                         + json.dumps({"diffs": rep["diffs"], "content": rep.get("content_problems")},
                                                      ensure_ascii=False)[:600])
                    acts.append("first_request_parity_passed")
            if spec.get('expected_first_request_sha256'):
                if spec.get('_first_request_validated') is False:
                    raise ValueError('frozen first-request gate failed earlier in this run; every request is refused')
                if spec.get('_first_request_validated') is None:
                    ok = request_digest(nd) == spec['expected_first_request_sha256']
                    spec['_first_request_validated'] = ok
                    if not ok:
                        raise ValueError('First model request differs from the frozen fake-resume request')
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
