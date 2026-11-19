"""Bind completed Chat Completions receipts to Pi session entries by upstream response ID."""
import json
import os
from pathlib import Path
import tempfile


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
    print(write_references(Path(sys.argv[1]), Path(sys.argv[2])))
