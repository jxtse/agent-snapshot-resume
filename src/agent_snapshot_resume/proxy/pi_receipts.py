"""Bind completed Chat Completions receipts to Pi session entries by upstream response ID."""
import json
import os
from pathlib import Path
import tempfile


def check_output(path):
    """Read this print invocation's NDJSON, independent of any seeded session history."""
    reason = None
    with path.open("rb") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                # Ordinary stderr lines may be interleaved; a broken JSON event is ambiguous.
                if line.lstrip().startswith(b"{"):
                    raise ValueError("malformed Pi JSON output")
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message") or {}
            if not isinstance(message, dict):
                raise ValueError("malformed Pi message event")
            if event.get("type") == "agent_end":
                messages = event.get("messages", [])
                if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
                    raise ValueError("malformed Pi agent_end event")
                assistants = [m for m in messages if m.get("role") == "assistant"]
                if assistants:
                    reason = assistants[-1].get("stopReason")
            if event.get("type") == "agent_start":
                reason = None
            if message.get("role") == "assistant":
                if event.get("type") == "message_start":
                    reason = None
                elif event.get("type") in ("message_end", "turn_end"):
                    reason = message.get("stopReason")
    if reason != "stop":
        raise ValueError(f"Pi final assistant stopReason: {reason!r}")


def compaction_kind(ref):
    if "pi_compaction_request" in ref:
        return ref["pi_compaction_request"]
    # Older receipts retained only item digests. Pi uses a dedicated stock system
    # message for its two-message auxiliary requests; never bind one to an assistant.
    try:
        from .native_wire import PI_SUMMARY_SYSTEM, digest
    except ImportError:  # standalone guest script
        from native_wire import PI_SUMMARY_SYSTEM, digest
    items = ref.get("items") or []
    if len(items) == 2 and "tools" not in (ref.get("fields") or {}):
        for role in ("system", "developer"):
            for content in (PI_SUMMARY_SYSTEM, [{"type": "text", "text": PI_SUMMARY_SYSTEM}]):
                if items[0] == digest({"role": role, "content": content}):
                    return "legacy"
    return None


def bind(session, requests):
    header, *entries = session
    if header.get("type") != "session" or header.get("version") != 3 or not header.get("id"):
        raise ValueError("invalid Pi session header")
    receipts = {}
    for row in requests:
        if row.get("path") != "/v1/chat/completions" or row.get("status") != 200:
            continue
        rid = row.get("response_id")
        if not rid:
            continue
        ref = row.get("native_reference") or {}
        if rid in receipts or ref.get("wire") != "chat" or ref.get("complete") is not True or not ref.get("raw_sha256"):
            raise ValueError("duplicate or invalid Chat Completions receipt")
        receipts[rid] = ref
    result, seen = [], set()
    for row in entries:
        msg = row.get("message") or {}
        if msg.get("role") != "assistant" or msg.get("stopReason") in {"error", "aborted"}:
            continue
        ref = receipts.pop(msg.get("responseId"), None)
        if ref is None or compaction_kind(ref) or not row.get("id") or row["id"] in seen:
            raise ValueError("missing or ambiguous Pi assistant receipt")
        seen.add(row["id"])
        result.append(dict(session_id=header["id"], entry_id=row["id"], response_id=msg["responseId"], content=ref))
    # Split-turn compaction can make two auxiliary calls for one session entry.
    # Never consume an unclassified leftover, or substitute a summary for an assistant.
    compactions = sum(r.get("type") == "compaction" for r in entries)
    auxiliary = [compaction_kind(ref) for ref in receipts.values()]
    if (not result or any(kind not in ("summary", "update", "turn_prefix", "legacy") for kind in auxiliary)
            or len(auxiliary) > 2 * compactions or auxiliary.count("turn_prefix") > compactions
            or auxiliary.count("summary") + auxiliary.count("update") > compactions):
        raise ValueError("session and completed receipts do not form a one-to-one match")
    return result


def write_references(session_path, log_dir):
    def rows(path):
        with path.open() as stream:
            yield from (json.loads(line) for line in stream if line.strip())
    records = bind(list(rows(session_path)), (r for p in sorted(log_dir.glob("requests-*.jsonl")) for r in rows(p)))
    with tempfile.NamedTemporaryFile(mode="w", dir=log_dir, delete=False) as out:
        try:
            for record in records:
                print(json.dumps(record, separators=(",", ":")), file=out)
            out.flush()
            os.replace(out.name, log_dir / "native-reference.jsonl")
        finally:
            Path(out.name).unlink(missing_ok=True)
    return len(records)


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--check-output":
        check_output(Path(sys.argv[2]))
    else:
        print(write_references(Path(sys.argv[1]), Path(sys.argv[2])))
