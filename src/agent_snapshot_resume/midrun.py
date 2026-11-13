"""Mid-run snapshots of a running resumed trial (host side; synchronous E2B SDK, runs in a thread).

A round-N+1 resume wants to start from the environment the failed round-N run had at its fork, not from the task
image replayed step by step. ``asr launch --midrun-snapshot-every`` therefore snapshots the *clone* while the agent
works: every N seconds (default 20 min) and after every context compaction, recording the snapshot id and the exact
session line that snapshot corresponds to:

    <runs-dir>/<job>/midrun-snapshots.jsonl
    {"ts", "t", "trial", "sandbox_id", "snapshot_id", "session_line", "session_line_before",
     "session_line_after", "session_path", "n_compactions", "trigger", "pause_s", "create_s"}

Facts this relies on (measured on Aliyun FC E2B, see the experiment log): ``create_snapshot`` takes ~7 s of host
time but suspends the guest only ~0.84 s, and a clone starts with the *processes* of the snapshotted guest still
running. So:

* a snapshot freezes the filesystem plus live processes; the session line count at the freeze is derived from the
  pause window (every session line carries a timestamp; writes during the pause are impossible).
* a clone taken from such a snapshot must have those processes stopped before its tail is replayed --
  ``sandbox.source.prepare_source(base_snapshot=...)`` does that.

The snapshotter never kills the clone: it only connects, reads, snapshots. It stops when the run dir stops being
written (the trial ended) or when ``stop()`` is called by the launcher.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

SANDBOX_RECORD = "midrun-sandbox.json"
RECORD_FILE = "midrun-snapshots.jsonl"
TICK_DIR = "/tmp/.asr-midrun"
TICK_FILE = TICK_DIR + "/ticks"

PROBE = r'''
import glob, json, os
def lines_of(p):
    n = 0
    comp = 0
    ts = []
    pending = []
    with open(p, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if b'"compact_boundary"' in line:
                comp += 1
            i = line.find(b'"timestamp":"')
            if i >= 0:
                ts.append(line[i + 13:i + 36].decode("ascii", "replace"))
            if b'"tool_use_id":"' in line and b'"type":"tool_result"' in line:
                j = line.find(b'"tool_use_id":"') + 15
                pending = [x for x in pending if x != line[j:j + 40]]
            if b'"type":"tool_use"' in line:
                j = line.find(b'"id":"') + 6
                pending.append(line[j:j + 40])
    return n, comp, ts, len(pending)
out = {"sessions": []}
for pat in ("/logs/agent/resume/sessions/projects/*/*.jsonl", "/logs/agent/sessions/projects/*/*.jsonl"):
    for p in sorted(glob.glob(pat)):
        n, comp, ts, pending = lines_of(p)
        out["sessions"].append({"path": p, "lines": n, "compactions": comp, "ts": ts, "pending_tools": pending,
                                "bytes": os.path.getsize(p)})
try:
    with open("/tmp/.asr-midrun/ticks") as f:
        # "monotonic epoch" per tick: the guest clock is what the session timestamps use
        out["ticks"] = [tuple(float(y) for y in x.split()) for x in f.read().splitlines()[-400:]]
except OSError:
    out["ticks"] = []
print(json.dumps(out))
'''

TICKER = (
    "import time\n"
    "f = open('/tmp/.asr-midrun/ticks', 'a', buffering=1)\n"
    "while True:\n"
    "    f.write('%.3f %.3f\\n' % (time.monotonic(), time.time()))\n"
    "    time.sleep(0.05)\n"
)


def _b64(script: str) -> str:
    return base64.b64encode(script.encode()).decode()


def parse_ts(ts: str) -> float | None:
    """Epoch seconds of a session line's `timestamp` (ISO8601, 'Z'), or None."""
    if not ts or len(ts) < 20:
        return None
    import datetime as dt
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def pause_window(before: list[Any], after: list[Any]) -> tuple[float, float] | None:
    """(start, end) of the suspension in guest *epoch* seconds, from the ticker values around the snapshot call."""
    if not before or not after:
        return None
    last_before, newest_after = before[-1][1], max(t[1] for t in after)
    gap = newest_after - last_before
    if gap <= 0.2:  # no suspension longer than two tick periods: cannot bound the freeze this way
        return None
    return last_before, newest_after


def session_line_at(probe: dict[str, Any], pause: tuple[float, float] | None) -> tuple[int, str, bool]:
    """(line count the snapshot contains, session path, reliable) for the resumed session.

    Every session line carries a timestamp in the guest clock, and the guest cannot write during the freeze, so the
    image holds exactly the lines written up to the pause start. If the pause cannot be measured the line count is
    unknown (``reliable`` false) and callers must not use this snapshot for a tail replay.
    """
    ses = max(probe.get("sessions") or [], key=lambda s: s["lines"])
    lines = int(ses["lines"])
    if pause is None:
        return lines, ses["path"], False
    keep = 0
    for ts in ses.get("ts") or []:
        t = parse_ts(ts)
        if t is not None and t <= pause[0]:
            keep += 1
    if not keep:
        return lines, ses["path"], False
    return min(lines, keep), ses["path"], True


def read_records(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


class Recorder:
    """Append-only record file; one line per snapshot, fsynced so a crashed launcher still leaves evidence."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict[str, Any]) -> dict[str, Any]:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t": round(time.time(), 3), **rec}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec


def sandbox_records(job_dir: Path) -> list[dict[str, Any]]:
    """Clones this job started, as written by SnapshotCloneEnvironment (`infra-validation/midrun-sandbox.json`)."""
    out = []
    for p in sorted(Path(job_dir).glob("*/infra-validation/" + SANDBOX_RECORD)):
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            continue
        rec["trial_dir"] = str(p.parent.parent)
        rec["trial"] = p.parent.parent.name
        out.append(rec)
    return out


class MidrunSnapshotter:
    """Snapshot every live clone of a job: periodic + after each compaction."""

    def __init__(self, job_dir: Path, record_path: Path | None = None, every_s: float = 1200.0,
                 poll_s: float = 60.0, quiet_after_s: float = 900.0, connect: Callable[[str], Any] | None = None,
                 log: Callable[[str], None] = print):
        self.job_dir = Path(job_dir)
        self.recorder = Recorder(record_path or (self.job_dir / RECORD_FILE))
        self.every_s = float(every_s)
        self.poll_s = float(poll_s)
        self.quiet_after_s = float(quiet_after_s)
        self._connect = connect
        self.log = log
        self._stop = threading.Event()
        self.state: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []

    # -- e2b
    def connect(self, sandbox_id: str):
        if self._connect is not None:
            return self._connect(sandbox_id)
        from e2b import Sandbox
        return Sandbox.connect(sandbox_id)

    def probe(self, sbx) -> dict[str, Any]:
        r = sbx.commands.run("echo " + _b64(PROBE) + " | base64 -d | python3 -", user="root", timeout=120)
        if r.exit_code != 0:
            raise RuntimeError("session probe failed: " + (r.stderr or "")[-400:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def ensure_ticker(self, sbx) -> None:
        r = sbx.commands.run("mkdir -p " + TICK_DIR + " && test -f " + TICK_FILE + " && echo running || echo absent",
                             user="root", timeout=60)
        if "running" in (r.stdout or ""):
            return
        sbx.commands.run("mkdir -p " + TICK_DIR + " && echo " + _b64(TICKER) + " | base64 -d > " + TICK_DIR +
                         "/tick.py && : > " + TICK_FILE, user="root", timeout=60)
        sbx.commands.run("nohup python3 " + TICK_DIR + "/tick.py >/dev/null 2>&1 &", user="root", timeout=0,
                         background=True)

    def snapshot(self, sbx, rec: dict[str, Any], trigger: str) -> dict[str, Any]:
        before = self.probe(sbx)
        t0 = time.time()
        snap = sbx.create_snapshot(request_timeout=300)
        create_s = round(time.time() - t0, 2)
        after = self.probe(sbx)
        pause = pause_window(before.get("ticks") or [], after.get("ticks") or [])
        line, path, measurable = session_line_at(after, pause)
        ses = max(after.get("sessions") or [], key=lambda s: s["lines"])
        before_ses = max(before.get("sessions") or [], key=lambda s: s["lines"], default=ses)
        # a tool call in flight at the freeze may have applied part of its effect already: replaying it would apply
        # that part twice, so such a snapshot is not a valid tail-replay base
        in_flight = int(ses.get("pending_tools") or 0)
        entry = {"trial": rec.get("trial"), "sandbox_id": rec.get("sandbox_id"), "snapshot_id": snap.snapshot_id,
                 "session_line": line, "session_line_before": before_ses["lines"],
                 "session_line_after": ses["lines"], "session_line_reliable": bool(measurable and not in_flight),
                 "in_flight_tools": in_flight,
                 "prefix_sha256": self.prefix_sha(sbx, path, line) if measurable else None,
                 "session_path": path, "n_compactions": before_ses["compactions"], "trigger": trigger,
                 "pause_s": round(pause[1] - pause[0], 3) if pause else None, "create_s": create_s,
                 "run_id": rec.get("run_id"), "session_id": rec.get("session_id")}
        return self.recorder.append(entry)

    def prefix_sha(self, sbx, path: str, line: int) -> str | None:
        """sha256 of the session prefix the image holds, so a later build can prove it resumes the same history."""
        if not path or line <= 0:
            return None
        try:
            r = sbx.commands.run(f"head -n {int(line)} {path} | sha256sum", user="root", timeout=120)
        except Exception:  # noqa: BLE001 - best effort: an unbound record is simply unusable for a tail replay
            return None
        out = (r.stdout or "").split()
        return out[0] if out and len(out[0]) == 64 else None

    # -- loop
    def tick(self) -> int:
        """One pass over the job's clones. Returns the number of snapshots taken."""
        n = 0
        for rec in sandbox_records(self.job_dir):
            sid = rec.get("sandbox_id")
            if not sid:
                continue
            st = self.state.setdefault(sid, {"last_t": 0.0, "compactions": 0, "done": False, "down": 0})
            if st["done"]:
                continue
            try:
                sbx = self.connect(sid)
            except Exception as exc:  # noqa: BLE001 - the clone may be gone (trial finished)
                st["down"] += 1
                if st["down"] >= 2:
                    st["done"] = True
                self.errors.append(f"{sid}: connect {exc!r}"[:300])
                continue
            try:
                self.ensure_ticker(sbx)
                probe = self.probe(sbx)
                ses = max(probe.get("sessions") or [], key=lambda s: s["lines"], default=None)
                if ses is None:
                    continue  # agent has not written its session yet
                comp = int(ses["compactions"])
                due_compaction = comp > st["compactions"]
                due_time = (time.time() - st["last_t"]) >= self.every_s
                if not (due_compaction or due_time or st["last_t"] == 0.0):
                    continue
                trigger = "start" if st["last_t"] == 0.0 else ("compaction" if due_compaction else "periodic")
                entry = self.snapshot(sbx, rec, trigger)
                st.update(last_t=time.time(), compactions=comp)
                n += 1
                self.log(json.dumps({"event": "midrun-snapshot", **{k: entry[k] for k in
                                                                    ("trial", "snapshot_id", "session_line",
                                                                     "n_compactions", "trigger", "pause_s",
                                                                     "create_s")}}, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001 - never kill the run because a snapshot failed
                self.errors.append(f"{sid}: snapshot {exc!r}"[:300])
                self.log(f"midrun-snapshot error {sid}: {exc!r}"[:400])
            finally:
                try:
                    sbx.disconnect() if hasattr(sbx, "disconnect") else None
                except Exception:  # noqa: BLE001
                    pass
        return n

    def run(self) -> None:
        idle_since: float | None = None
        while not self._stop.is_set():
            try:
                took = self.tick()
            except Exception as exc:  # noqa: BLE001
                self.errors.append(repr(exc)[:300])
                took = 0
            # stop once the job has produced no trial dirs for a while (the launcher normally stops us explicitly)
            have = bool(self.job_dir.exists() and any(self.job_dir.iterdir())) if self.job_dir.exists() else False
            idle_since = None if have else (idle_since or time.time())
            if idle_since and time.time() - idle_since > self.quiet_after_s:
                return
            self._stop.wait(self.poll_s)

    def stop(self) -> None:
        self._stop.set()

    def start_thread(self) -> threading.Thread:
        th = threading.Thread(target=self.run, name="midrun-snapshots", daemon=True)
        th.start()
        return th
