"""List / kill sandboxes that carry this tool's owner tag, and delete snapshots.

Only sandboxes whose metadata `owner` equals ASR_OWNER_TAG are listed; --kill-id refuses other sandboxes unless
--force is given.
"""
from __future__ import annotations

import time

from ..config import owner_tag

PEEK_CMD = r"""
for f in /logs/agent/resume-proxy/*/requests.jsonl; do [ -f "$f" ] && echo "## $f $(wc -l < $f)" && tail -2 $f | cut -c1-400; done
for f in /logs/agent/model-proxy/requests-*.jsonl; do [ -f "$f" ] && echo "## $f $(wc -l < $f)" && tail -1 $f | cut -c1-300; done
date -u +%FT%TZ
"""


def list_own(owner: str | None = None):
    from e2b import Sandbox
    from e2b.sandbox.sandbox_api import SandboxQuery

    owner = owner or owner_tag()
    for _ in range(6):  # Sandbox.list() can raise ValueError while any sandbox on the account is snapshotting
        try:
            out, pag = [], Sandbox.list(query=SandboxQuery(metadata={"owner": owner}))
            while pag.has_next:
                out += pag.next_items()
            return out
        except ValueError as e:
            if "snapshotting" not in str(e):
                raise
            time.sleep(5)
    raise RuntimeError("Sandbox.list kept failing with the snapshotting state")


def kill(sandbox_id: str, force: bool = False) -> bool:
    from e2b import Sandbox

    if not force:
        info = Sandbox.get_info(sandbox_id)
        if (info.metadata or {}).get("owner") != owner_tag():
            raise RuntimeError(f"{sandbox_id} is not tagged owner={owner_tag()}; use --force to kill anyway")
    return Sandbox.kill(sandbox_id)


def peek(sandbox_id: str) -> str:
    from e2b import Sandbox

    return Sandbox.connect(sandbox_id).commands.run(PEEK_CMD, user="root", timeout=60).stdout


def delete_snapshot(snapshot_id: str) -> bool:
    from e2b import Sandbox

    return Sandbox.delete_snapshot(snapshot_id)
