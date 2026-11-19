"""JSON request contracts for Responses and OpenAI-compatible Chat Completions.

Standalone (uploaded alongside resume_proxy.py). References contain hashes, never
prompt excerpts. An incomplete historical reference can disprove parity only.
"""
from datetime import date
import hashlib
import json
import re
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

FIELDS = {"responses": "input", "chat": "messages"}
PATHS = {"responses": "v1/responses", "chat": "v1/chat/completions"}


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def body_reference(wire, body):
    history = FIELDS[wire]
    items = body[history]
    if not isinstance(items, list):
        raise ValueError(f"{history} must be an array")
    return {"wire": wire, "items": [digest(i) for i in items], "complete": True,
            "fields": {k: digest(v) for k, v in body.items() if k != history}}


def codex_item(item):
    return {k: v for k, v in item.items() if k not in {"id", "internal_chat_message_metadata_passthrough"}
            and v is not None}


def user_item(wire, text):
    if wire == "chat":
        return {"role": "user", "content": text}
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


# Digests from the stock release with the single system-skills root parameterized.
SKILLS_SHA = "92a6b3b09dc0af7ed18330912c8962046758c43d034de484b454336551db944e"
PERMISSIONS_SHA = "29dad5ed6993c5e717376e0a1c84d54ab60ee593510916ac6cee4295e803315d"
ENVIRONMENT_SHA = "f99aa2b0f07117d1a8cdde90f6ae1c14e0f573cdaedfaa2495ce97b3f9fd089c"
# danger-full-access, resumed in the recorded cwd/shell: Codex 0.154.0 omits <cwd> and <shell> (observed on Linux).
ENVIRONMENT_UNCHANGED_CWD_SHA = "0fc170068deac9c68bf72770ea8e04def06805cebf44801040b560478dc8a36a"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resume_blocks(item, role, count):
    if (set(item) != {"type", "role", "id", "content"} or item["type"] != "message" or item["role"] != role
            or not isinstance(item["id"], str)
            or not re.fullmatch(r"msg_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", item["id"])):
        raise ValueError(f"unexpected Codex resume {role} item")
    blocks = item["content"]
    if (not isinstance(blocks, list) or len(blocks) != count
            or any(not isinstance(b, dict) or set(b) != {"type", "text"} or b["type"] != "input_text"
                   or not isinstance(b["text"], str) for b in blocks)):
        raise ValueError(f"unexpected Codex resume {role} content")
    return [b["text"] for b in blocks]


def check_codex_context(developer, user, artifacts):
    mode = artifacts.get("sandbox_mode")
    if mode == "danger-full-access":
        # With the original permissions restored, the CLI only adds the system skills here.
        skills, = resume_blocks(developer, "developer", 1)
        environment_hash = "746971acea2e78edbf33170ee8de6165122f163d5f324d2bc91fd6ed5595531f"
    elif mode in (None, "read-only"):
        skills, permissions = resume_blocks(developer, "developer", 2)
        if text_hash(permissions) != PERMISSIONS_SHA:
            raise ValueError("unrecognized Codex resume permissions")
        environment_hash = ENVIRONMENT_SHA
    else:
        raise ValueError("unsupported Codex resume sandbox mode")
    env, = resume_blocks(user, "user", 1)
    home, cwd = artifacts.get("codex_home"), artifacts.get("cwd")
    if not all(isinstance(p, str) and p.startswith("/") for p in (home, cwd)):
        raise ValueError("Codex resume paths must be absolute")
    root = f"`{home}/skills/.system`"
    if skills.count(root) != 1:
        raise ValueError("unexpected Codex resume skills root")
    skills = skills.replace(root, "`{CODEX_HOME}/skills/.system`", 1)
    if text_hash(skills) != SKILLS_SHA:
        raise ValueError("unrecognized Codex resume skills or permissions")
    # Codex sends the environment context as a diff against the session's last one: when the resumed cwd and shell
    # equal the recorded ones (the faithful sandbox case), <cwd> and <shell> are omitted. Both tags must then be
    # absent together, and the template hash below differs accordingly.
    unchanged_cwd = "<cwd>" not in env and "<shell>" not in env and mode == "danger-full-access"
    if unchanged_cwd:
        environment_hash = ENVIRONMENT_UNCHANGED_CWD_SHA
    for tag, pattern in (("cwd", re.escape(escape(cwd))), ("root", re.escape(escape(cwd))),
                         ("shell", r"(?:ba|z|fi|da)?sh"), ("current_date", r"[0-9]{4}-[0-9]{2}-[0-9]{2}"),
                         ("timezone", r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*")):
        if unchanged_cwd and tag in ("cwd", "shell"):
            continue
        match = re.search(fr"<{tag}>({pattern})</{tag}>", env)
        if not match:
            raise ValueError(f"unexpected Codex resume environment {tag}")
        if tag == "current_date":
            date.fromisoformat(match[1])
        elif tag == "timezone":
            try:
                ZoneInfo(match[1])
            except ZoneInfoNotFoundError:
                raise ValueError("unexpected Codex resume timezone") from None
        env = env[:match.start(1)] + "{" + tag + "}" + env[match.end(1):]
    if text_hash(env) != environment_hash:
        raise ValueError("unrecognized Codex resume environment or filesystem permissions")



# Exact stock Pi 1.4.0 prompts from core/compaction/{compaction,utils}.js.
PI_SUMMARY_SYSTEM = 'You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.'
PI_INSTRUCTIONS = {'summary': '9b00aa68df1a64279bc36e9093367f638701d48ec82e3d08436f65092a515f9b', 'update': '240c52982209146eae47d73c7172f6ba1dab60f44520bcb6a6ec1a883fef2ec7', 'turn_prefix': '7b0a15be5e0fdcce338ce7c1ef1fdd386faed258cc25d03915a05b05e7a6a183'}


def single_text(content):
    if isinstance(content, str):
        return content
    if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
            and set(content[0]) == {"type", "text"} and content[0]["type"] == "text"):
        return content[0]["text"]
    return None


def pi_summary_parts(body):
    """Return the serialized conversation and its exact wrapper for a pinned summary request."""
    messages = body.get("messages") or []
    if (not isinstance(messages, list) or len(messages) != 2 or not all(isinstance(m, dict) for m in messages)
            or messages[0].get("role") not in ("system", "developer") or messages[1].get("role") != "user"
            or body.get("tools")):
        return None
    if single_text(messages[0].get("content")) != PI_SUMMARY_SYSTEM:
        return None
    prompt = single_text(messages[1].get("content"))
    if not isinstance(prompt, str):
        return None
    for opening, closing, kind in (("<conversation>\n", "\n</conversation>\n\n", "summary"),
                                   ("# Conversation\n", "\n\n# Instructions\n", "turn_prefix")):
        if not prompt.startswith(opening) or prompt.count(closing) != 1:
            continue
        transcript, suffix = prompt[len(opening):].split(closing)
        instructions = suffix
        if kind == "summary" and suffix.startswith("<previous-summary>\n"):
            if suffix.count("\n</previous-summary>\n\n") != 1:
                return None
            instructions = suffix.split("\n</previous-summary>\n\n")[1]
            kind = "update"
        instructions = instructions.partition("\n\nAdditional focus: ")[0]
        if text_hash(instructions) == PI_INSTRUCTIONS[kind]:
            return opening, transcript, closing + suffix, kind
    return None


def native_request_metrics(body, spec):
    wire = spec["wire"]
    items = body[FIELDS[wire]]
    parts = pi_summary_parts(body) if wire == "chat" else None
    hints = list(spec.get("prior_hints") or [])
    if spec.get("mode") == "hint" and spec.get("hint"):
        hints.append(spec["hint"])
    counts = []
    for hint in hints:
        if parts:
            text = parts[1]
            marker = "[User]: " + hint
            matches = re.findall(r"(?:\A|\n\n)" + re.escape(marker) + r"(?=\Z|\n\n\[(?:User|Assistant|Assistant tool calls|Tool result)\]: )", text)
            count = len(matches)
        else:
            count = sum(i == user_item(wire, hint) or (i.get("role") == "user" and single_text(i.get("content")) == hint)
                        for i in items)
        counts.append(count)
    return {"wire": wire, "n_items": len(items), "hint_blocks": counts[-1] if counts else 0,
            "hint_chain": [{"blocks": count} for count in counts],
            "compaction_request": parts[3] if parts else None}



def rewrite_native(body, spec, *, first=False):
    wire = spec["wire"]
    field = FIELDS[wire]
    history = body.get(field)
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise ValueError(f"expected {field} array")
    sentinel = spec["sentinel"]

    def exact(item):
        if item.get("role") != "user" or item.get("type", "message") != "message":
            return False
        content = item.get("content")
        return content == sentinel or content == [{"type": "text" if wire == "chat" else "input_text", "text": sentinel}]

    parts = pi_summary_parts(body) if wire == "chat" and not first else None
    if parts and sentinel in parts[1]:
        opening, transcript, suffix, kind = parts
        turns = transcript.split("\n\n")
        marker = "[User]: " + sentinel
        if turns.count(marker) != 1 or json.dumps(body, ensure_ascii=False).count(sentinel) != 1:
            raise ValueError("unexpected Pi summary sentinel position")
        index = turns.index(marker)
        if ((index and not re.match(r"^\[(?:User|Assistant|Assistant tool calls|Tool result)\]: ", turns[0]))
                or index + 1 < len(turns) and not re.match(r"^\[(?:User|Assistant|Assistant tool calls|Tool result)\]: ", turns[index+1])):
            raise ValueError("unexpected Pi summary turn boundary")
        hint = spec.get("mode") == "hint"
        turns[index:index+1] = ["[User]: " + spec["hint"]] if hint else []
        prompt = opening + "\n\n".join(turns) + suffix
        content = prompt if isinstance(history[-1]["content"], str) else [{"type": "text", "text": prompt}]
        return {**body, field: [history[0], {**history[1], "content": content}]}, [
            "replace_summary_sentinel" if hint else "drop_summary_sentinel"]
    hits = [i for i, item in enumerate(history) if exact(item)]
    if any(sentinel in json.dumps(item, ensure_ascii=False) for item in history if not exact(item)):
        raise ValueError("resume sentinel appears outside an exact user turn")
    if not hits and not first and not spec.get("require_sentinel"):
        return body, ["sentinel_not_found"]
    if len(hits) != 1:
        raise ValueError("expected exactly one native resume sentinel")
    index = hits[0]
    hint = [user_item(wire, spec["hint"])] if spec.get("mode") == "hint" else []
    start = index
    actions = []
    artifacts = spec.get("resume_artifacts") or {}
    if "resume_context" in artifacts:
        if wire != "responses" or artifacts["resume_context"] != "codex-0.154.0-skills-permissions-environment":
            raise ValueError("unknown native resume context")
        if index < 2 or (first and index != len(history) - 1):
            raise ValueError("unexpected Codex resume context position")
        if first and "prefix_items" in artifacts and index - 2 != artifacts["prefix_items"]:
            raise ValueError("Codex resume context is not at the referenced cut")
        if resume_blocks(history[index], "user", 1) != [sentinel]:
            raise ValueError("unexpected Codex resume prompt")
        check_codex_context(history[index-2], history[index-1], artifacts)
        start -= 2
        actions.extend(["drop_resume_developer", "drop_resume_environment"])
    actions.append("replace_sentinel" if hint else "drop_sentinel")
    return {**body, field: history[:start] + hint + history[index + 1:]}, actions


def compare_native(body, ref, hints=()):
    wire = ref["wire"]
    history = body.get(FIELDS[wire])
    problems = []
    if not isinstance(history, list) or not all(isinstance(i, dict) for i in history):
        return {"ok": False, "verdict": "mismatch", "diffs": {}, "all_items_equal": False,
                "evidence_complete": ref["complete"], "content_problems": [f"invalid {FIELDS[wire]} array"]}
    history = list(history)
    for hint in reversed(hints):
        if not history or history[-1] != user_item(wire, hint):
            problems.append("missing exact hint at cut")
            break
        history.pop()
    if ref.get("projection") == "codex_rollout":
        history = [codex_item(item) for item in history]
    got = body_reference(wire, {**body, FIELDS[wire]: history})
    unequal = [i for i, (actual, expected) in enumerate(zip(got["items"], ref["items"])) if actual != expected]
    if unequal:
        problems.append(f"{FIELDS[wire]}[{unequal[0]}] digest differs")
    if ref.get("logged_item_count") is not None and len(history) != ref["logged_item_count"]:
        problems.append(f"{FIELDS[wire]} length differs from historical n_items")
    if len(history) != len(ref["items"]):
        problems.append(f"{FIELDS[wire]} length {len(history)} != {len(ref['items'])}")
    if ref.get("raw_sha256") and not problems:
        candidate = {**body, FIELDS[wire]: history}
        raw = json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode()
        if hashlib.sha256(raw).hexdigest() != ref["raw_sha256"]:
            problems.append("serialized request bytes differ")
    fields = set(ref["fields"])
    if ref["complete"]:
        fields.update(got["fields"])
    for key in sorted(fields):
        if got["fields"].get(key) != ref["fields"].get(key):
            problems.append(f"{key} digest differs")
    verdict = "mismatch" if problems else ("ok" if ref["complete"] else "incomplete_evidence")
    return {"ok": verdict == "ok", "verdict": verdict, "evidence_complete": ref["complete"],
            "all_items_equal": not unequal and len(history) == len(ref["items"]),
            "item_count": len(history), "expected_item_count": len(ref["items"]), "unequal_indices": unequal,
            "diffs": {}, "content_problems": problems}


def envelope_only_gap(ref):
    return (ref.get("wire") == "responses" and ref.get("complete") is False
            and ref.get("projection") == "codex_rollout_tokens"
            and ref.get("missing_evidence") == ["original_request_envelope"]
            and bool(ref.get("items")) and len(ref["items"]) == ref.get("logged_item_count")
            and {"model", "instructions"} <= ref.get("fields", {}).keys())


def evaluate_native_gate(body, exp, hints, allow, translate=None):
    ref = exp["content"]
    report = compare_native(body, ref, hints)
    report.update(allow_incomplete_evidence=allow, admitted=report["ok"], freeze_sha256=digest(body))
    if not (allow and report["verdict"] == "incomplete_evidence" and envelope_only_gap(ref)):
        return report
    metadata = exp.get("metadata") or {}
    if metadata:
        if translate is None:
            report["content_problems"].append("no pinned translator to verify historical request metrics")
            return report
        stripped = dict(body, input=body["input"][:-len(hints)]) if hints else body
        try:
            payload, _, details = translate(stripped)
        except Exception as exc:
            raise ValueError("pinned translator rejected the request") from exc
        metrics = dict(details, req_bytes=len(json.dumps(payload)))
        report["translator_metrics"] = {k: metrics.get(k) for k in metadata}
        report["translator_diffs"] = {k: [v, metrics.get(k)] for k, v in metadata.items() if metrics.get(k) != v}
        if report["translator_diffs"]:
            report.update(verdict="mismatch")
            return report
        # The translator discards per-process tracing metadata. Bind everything else at both interfaces.
        if "client_metadata" in details.get("dropped_fields", ()):
            actual_payload = translate(body)[0] if hints else payload
            report["translated_request_sha256"] = digest(actual_payload)
            report["freeze_sha256"] = digest([{k: v for k, v in body.items() if k != "client_metadata"},
                                              report["translated_request_sha256"]])
    report["admitted"] = True
    return report
