"""Context-equivalence check: rewritten first request vs the original session prefix (Claude Code sessions).

Reference = the original session lines[:cut] (exactly what the original model had seen when it produced step k,
as persisted by Claude Code). Every model-visible content block persisted in that prefix (instruction text, assistant
thinking text+signature, assistant text, tool_use id/name/input, tool_result id/is_error/content) must appear in the
request in the same order and byte-identical; anything else in the request is listed as 'extra' and classified.
Usage: asr equiv <session.jsonl> <cut_line> <first_request_rewritten.json> [--sentinel S] [--hint TEXT] [--out report.json]
"""
import argparse, json, re, sys

CONTINUE_TXT = "Continue from where you left off."
SYNTH_TXT = "No response requested."


def _norm_tr_content(c):
    if isinstance(c, str):
        return c
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def blocks_from_session(lines):
    """CC persists one line per content block (parallel tool calls interleave tool_use/tool_result lines);
    the API request merges them per assistant message id: [all assistant blocks] then [all tool_results]."""
    out, a_part, r_part, cur = [], [], [], None

    def flush():
        out.extend(a_part)
        out.extend(r_part)
        a_part.clear()
        r_part.clear()

    for d in lines:
        t = d.get("type")
        if t not in ("user", "assistant") or d.get("isMeta"):
            continue
        c = d["message"]["content"]
        role = d["message"]["role"]
        if t == "assistant":
            mid = d["message"].get("id")
            if mid != cur:
                flush()
                cur = mid
            _emit(a_part, role, c)
        elif isinstance(c, list) and c and all(b.get("type") == "tool_result" for b in c):
            _emit(r_part, role, c)
        else:
            flush()
            cur = None
            _emit(out, role, c)
    flush()
    return out


def _emit(out, role, c):
    if isinstance(c, str):
        out.append((role, "text", c))
        return
    for b in c:
        bt = b.get("type")
        if bt == "text":
            if role == "assistant" and not b.get("text"):
                continue
            out.append((role, "text", b["text"]))
        elif bt == "thinking":
            out.append((role, "thinking", b.get("thinking", ""), b.get("signature")))
        elif bt == "tool_use":
            out.append((role, "tool_use", b["id"], b["name"], json.dumps(b.get("input"), sort_keys=True, ensure_ascii=False)))
        elif bt == "tool_result":
            out.append((role, "tool_result", b["tool_use_id"], bool(b.get("is_error")), _norm_tr_content(b.get("content"))))
        else:
            out.append((role, bt, json.dumps(b, sort_keys=True)))
    return out


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
                out.append(((role, "tool_use", b["id"], b["name"], json.dumps(b.get("input"), sort_keys=True, ensure_ascii=False)), i))
            elif bt == "tool_result":
                out.append(((role, "tool_result", b["tool_use_id"], bool(b.get("is_error")), _norm_tr_content(b.get("content"))), i))
            else:
                out.append(((role, bt, json.dumps({k: v for k, v in b.items() if k != "cache_control"}, sort_keys=True)), i))
    return out


_REMINDER_SUFFIX_RE = re.compile(r"(?:\n\n<system-reminder>\n.*?\n</system-reminder>)+", re.S)


def _reminder_suffixed(blk, ref_blk):
    """True if blk is ref_blk's tool_result with only <system-reminder> blocks appended to its string content."""
    if blk[:4] != ref_blk[:4] or blk[1] != "tool_result":
        return False
    got, want = blk[4], ref_blk[4]
    return got.startswith(want) and bool(_REMINDER_SUFFIX_RE.fullmatch(got[len(want):]))


def classify_extra(blk):
    role, bt = blk[0], blk[1]
    txt = blk[2] if bt == "text" else ""
    if role == "system":
        return "cc_system_message(agent/skill listing)"
    if "<system-reminder>" in txt and "currentDate" in txt:
        return "cc_currentDate_reminder"
    if "<task-notification>" in txt:
        return "background_task_notification(queued_command attachment)"
    if "<system-reminder>" in txt:
        return "system_reminder(attachment-derived)"
    if txt.strip() == CONTINUE_TXT or txt.strip() == SYNTH_TXT:
        return "RESUME_ARTIFACT"
    return "other"


def context_lines(lines):
    """The session lines that form the model context, as Claude Code rebuilds it on --resume.

    Without compaction: all lines. After the last `compact_boundary` system line: the lines after the boundary up to
    and including the compact-summary user message, then the preserved segment (compactMetadata.preservedMessages:
    the last messages before the boundary, kept verbatim), then the lines after the summary.
    """
    b = None
    for i, d in enumerate(lines):
        if d.get("type") == "system" and d.get("subtype") == "compact_boundary":
            b = i
    if b is None:
        return lines
    post = lines[b + 1:]
    s = next((i for i, d in enumerate(post) if d.get("type") == "user" and d.get("isCompactSummary")), None)
    if s is None:
        raise ValueError(f"compact_boundary at line {b} has no compact-summary user message after it")
    meta = lines[b].get("compactMetadata") or {}
    keep = (meta.get("preservedMessages") or {}).get("uuids") or []
    by_uuid = {d.get("uuid"): d for d in lines[:b] if d.get("uuid")}
    missing = [u for u in keep if u not in by_uuid]
    if missing:
        raise ValueError(f"compact_boundary at line {b}: preserved messages not found before it: {missing[:3]}")
    return post[:s + 1] + [by_uuid[u] for u in keep] + post[s + 1:]


def check(session_path, cut, req_path, hint=None, sentinel=None):
    with open(session_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    ref = blocks_from_session(context_lines(lines[:cut]))
    with open(req_path) as f:
        rq = json.load(f)
    body = rq.get("body", rq)
    msgs = body["messages"]
    got = blocks_from_request(msgs)
    # in-order subsequence match of ref within got
    j = 0
    matched = []
    extras = []
    mism = []
    full_text = []  # full text of each extra block (the preview is truncated; hints can be longer than it)
    reminder_suffixes = []
    for blk, mi in got:
        if j < len(ref) and blk == ref[j]:
            matched.append(mi)
            j += 1
        elif j < len(ref) and _reminder_suffixed(blk, ref[j]):
            # Claude Code renders some persisted attachments (e.g. after a compaction) as <system-reminder> text
            # appended to the adjacent tool_result content; the persisted result itself is unchanged.
            reminder_suffixes.append({"msg_index": mi, "tool_use_id": blk[2],
                                      "suffix_preview": blk[4][len(ref[j][4]):][:300]})
            matched.append(mi)
            j += 1
        else:
            text = blk[2] if isinstance(blk[2], str) else str(blk[2])
            full_text.append(text)
            extras.append({"msg_index": mi, "role": blk[0], "type": blk[1], "class": classify_extra(blk),
                           "preview": text[:300]})
    missing = ref[j:]
    if missing:
        nxt = missing[0]
        mism.append({"first_missing_ref_index": j, "role": nxt[0], "type": nxt[1], "preview": str(nxt[2])[:300]})
    artifacts = [e for e in extras if e["class"] == "RESUME_ARTIFACT"]
    sent_left = bool(sentinel) and any(sentinel in json.dumps(m, ensure_ascii=False) for m in msgs)
    hint_blocks = [e for e, text in zip(extras, full_text) if hint in text] if hint else []
    last = msgs[-1]
    last_is_tr = last["role"] == "user" and any(isinstance(last["content"], list) and b.get("type") == "tool_result" for b in last["content"])
    ok = (not missing and not artifacts and not sent_left and (len(hint_blocks) == 1 if hint else True))
    # attachment correspondence: CC renders persisted attachments as system messages / reminders
    att = [d["attachment"].get("type") for d in context_lines(lines[:cut]) if d.get("type") == "attachment"]
    rendered_att = [e for e, text in zip(extras, full_text)
                    if e["class"] not in ("cc_currentDate_reminder", "RESUME_ARTIFACT") and not (hint and hint in text)]
    att_ok = len(rendered_att) + len(reminder_suffixes) <= len(att)
    ok = ok and att_ok
    return {
        "equivalent": ok, "n_ref_blocks": len(ref), "n_request_blocks": len(got), "n_matched_in_order": j,
        "n_missing": len(missing), "first_mismatch": mism, "n_extra": len(extras),
        "extra_by_class": _count(extras), "extras": extras[:50], "resume_artifacts_left": artifacts,
        "sentinel_left": sent_left, "hint_blocks": len(hint_blocks), "last_message_is_tool_result": last_is_tr,
        "prefix_attachments": att, "n_rendered_extra_blocks": len(rendered_att), "attachment_correspondence_ok": att_ok,
        "tool_result_reminder_suffixes": reminder_suffixes,
        "request_meta": {"model": body.get("model"), "n_messages": len(msgs), "n_tools": len(body.get("tools") or []),
                         "tools": [t.get("name") for t in body.get("tools") or []], "thinking": body.get("thinking"),
                         "max_tokens": body.get("max_tokens"), "output_config": body.get("output_config"),
                         "system_chars": sum(len(b.get("text", "")) for b in body.get("system") or []) if isinstance(body.get("system"), list) else len(body.get("system") or "")},
    }


def _count(xs):
    c = {}
    for e in xs:
        c[e["class"]] = c.get(e["class"], 0) + 1
    return c


def build_parser(ap=None):
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session")
    ap.add_argument("cut", type=int)
    ap.add_argument("request")
    ap.add_argument("--hint")
    ap.add_argument("--sentinel")
    ap.add_argument("--out")
    return ap


def main(a=None):
    a = a or build_parser().parse_args()
    r = check(a.session, a.cut, a.request, a.hint, a.sentinel)
    s = json.dumps(r, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    print(json.dumps({k: r[k] for k in ("equivalent", "n_ref_blocks", "n_matched_in_order", "n_missing", "extra_by_class", "sentinel_left", "hint_blocks", "last_message_is_tool_result")}))
    return 0 if r["equivalent"] else 1


if __name__ == "__main__":
    sys.exit(main())
