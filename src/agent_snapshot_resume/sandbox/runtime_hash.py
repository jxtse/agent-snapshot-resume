"""Runtime file-hash collection for the restored-environment gates, fail-closed but never fatal.

The gates hash files *inside* a sandbox and compare them with a host-side expectation (the task's ``tests/`` for the
verifier, the bundle's auxiliary files for a clone). Collection used to be a one-liner that called
``pathlib.Path(p).read_bytes()`` for every path: one unreadable path (missing, a directory, a socket/FIFO, a dangling
symlink, a permission error) raised out of the collection loop and the whole check died with a traceback taken from
the *guest*, which read like an infrastructure crash and hid which file was actually wrong. A trial then never reached
its verifier even though the agent's work was finished.

Collection is now per path: every path yields either a sha256 or a recorded reason, the reasons are reported with the
file names, and the check raises one readable error. A verifier environment whose image ships an *incomplete*
``/tests`` (only the top-level test files, no nested data dirs) is repaired from the task's own ``tests/`` and then
re-checked - see ``SnapshotCloneEnvironment._repair_verifier_tests``.
"""
from __future__ import annotations

import io
import json
import shlex
import tarfile
from pathlib import Path

# Guest-side collector: one JSON object per requested path, never an exception. Chunked hashing keeps a 16 MB
# volume out of the guest's memory budget.
COLLECT_SCRIPT = """import hashlib, json, os, stat, sys
out = {}
for p in json.loads(sys.argv[1]):
    try:
        st = os.stat(p)
    except OSError as exc:
        out[p] = {"error": type(exc).__name__, "detail": str(exc)[:300]}
        continue
    if stat.S_IFMT(st.st_mode) != stat.S_IFREG:
        out[p] = {"error": "NotARegularFile", "detail": oct(stat.S_IFMT(st.st_mode))}
        continue
    digest = hashlib.sha256()
    try:
        with open(p, "rb", buffering=0) as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        out[p] = {"error": type(exc).__name__, "detail": str(exc)[:300]}
        continue
    out[p] = {"sha256": digest.hexdigest(), "size": st.st_size}
print(json.dumps(out))
"""


def collect_command(paths) -> str:
    """The guest command that collects the hashes of `paths` (one JSON line on stdout)."""
    return "python3 -c " + shlex.quote(COLLECT_SCRIPT) + " " + shlex.quote(json.dumps(list(paths)))


def parse_collect(stdout: str) -> tuple[dict[str, str], dict[str, str]]:
    """(hashes, errors) from the collector's stdout. `errors` maps path -> "Kind: detail"."""
    raw = json.loads(stdout.strip() or "{}")
    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    for path, value in raw.items():
        if isinstance(value, dict) and value.get("sha256"):
            hashes[path] = value["sha256"]
        else:
            detail = value if not isinstance(value, dict) else {k: v for k, v in value.items() if k != "sha256"}
            kind = detail.get("error") if isinstance(detail, dict) else None
            note = detail.get("detail") if isinstance(detail, dict) else str(detail)
            errors[path] = f"{kind or 'error'}: {note}" if note else (kind or "error")
    return hashes, errors


def hash_problems(expected: dict[str, str], actual: dict[str, str], errors: dict[str, str]) -> list[str]:
    """Everything that keeps `expected` from being matched, one readable line per file (empty == all good)."""
    out = []
    for path in sorted(expected):
        want = expected[path]
        if path in errors:
            out.append(f"{path}: unreadable ({errors[path]})")
        elif path not in actual:
            out.append(f"{path}: not hashed")
        elif actual[path] != want:
            out.append(f"{path}: sha256 differs (expected {want[:12]}..., got {actual[path][:12]}...)")
    return out


def tests_files_tar_bytes(src: Path, include) -> tuple[bytes, int]:
    """A gzipped tar of the named files of a task's ``tests/``, for restoring a partial verifier image.

    Only the paths the gate expects travel (the expected-hash keys, relative to /tests): a task image may
    deliberately *not* carry some of ``tests/`` (e.g. the packed population archives a build step unpacks and drops),
    and adding those back would make the task's own corpus check fail on "extra" files. Modes travel with the
    archive (``test.sh`` stays executable); a name missing locally is skipped (the hash check then reports it).
    """
    src = Path(src)
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in sorted(include):
            path = src / rel
            if not path.is_file():
                continue
            tar.add(path, arcname=rel, recursive=False)
            n += 1
    return buf.getvalue(), n
