"""Mid-run snapshotter: record bookkeeping, the pause window / session-line derivation, and the tick loop
(driven by a fake sandbox: no E2B, no credentials)."""
import json
import tempfile
import unittest
from pathlib import Path

from agent_snapshot_resume import midrun


class Run:
    def __init__(self, stdout="", exit_code=0):
        self.stdout = stdout
        self.stderr = ""
        self.exit_code = exit_code


class FakeSandbox:
    def __init__(self, probes, snapshot_ids=None):
        self.probes = list(probes)
        self.snapshot_ids = list(snapshot_ids or [f"snap-{i}" for i in range(1, 10)])
        self.snapshots = 0
        self.commands = self

    def run(self, cmd, **kw):
        if "python3" in cmd and "base64 -d" in cmd:
            return Run(json.dumps(self.probes.pop(0) if len(self.probes) > 1 else self.probes[0]))
        if "echo running" in cmd:
            return Run("running\n")
        return Run("")

    def create_snapshot(self, request_timeout=300):
        self.snapshots += 1
        sid = self.snapshot_ids.pop(0) if self.snapshot_ids else f"snap-{self.snapshots}"

        class Snap:
            snapshot_id = sid
            names = []
        return Snap()


def probe(lines, compactions=0, ts=None, ticks=None, pending_tools=0):
    return {"sessions": [{"path": "/logs/agent/resume/sessions/projects/-app/s.jsonl", "lines": lines,
                          "compactions": compactions, "bytes": 10, "ts": ts or [],
                          "pending_tools": pending_tools}],
            "ticks": ticks or []}


T0 = midrun.parse_ts("2026-01-01T00:00:00.000Z")


def ticks(*epochs):
    """Ticker values as the guest writes them: (monotonic, epoch)."""
    return [(e - T0, e) for e in epochs]


class PauseWindow(unittest.TestCase):
    def test_gap_between_last_tick_before_and_first_after(self):
        # the guest clock is what matters: the pause is measured on the epoch part
        self.assertEqual(midrun.pause_window(ticks(T0 + 1.0, T0 + 1.05, T0 + 1.1), ticks(T0 + 1.95, T0 + 2.0)),
                         (T0 + 1.1, T0 + 2.0))
        self.assertIsNone(midrun.pause_window(ticks(T0 + 1.0), ticks(T0 + 1.05)))  # not a suspension
        self.assertIsNone(midrun.pause_window([], ticks(T0 + 1.0)))

    def test_session_line_is_the_state_at_the_pause(self):
        ts = ["2026-01-01T00:00:00.000Z"] * 5 + ["2026-01-01T00:10:00.000Z"] * 3
        # five lines written before the pause; the three written after it are not in the image
        pause = (midrun.parse_ts("2026-01-01T00:05:00.000Z"), midrun.parse_ts("2026-01-01T00:09:00.000Z"))
        line, path, reliable = midrun.session_line_at(probe(8, ts=ts), pause)
        self.assertEqual((line, reliable), (5, True))
        self.assertTrue(path.endswith(".jsonl"))

    def test_unmeasurable_pause_is_flagged_unreliable(self):
        ts = ["2026-01-01T00:00:00.000Z"] * 8
        line, _path, reliable = midrun.session_line_at(probe(8, ts=ts), None)
        self.assertFalse(reliable)
        line, _path, reliable = midrun.session_line_at(probe(8, ts=ts, ticks=ticks(1.0)), None)
        self.assertFalse(reliable)


class TickLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job = self.root / "jobs" / "j1"
        trial = self.job / "task__abc"
        (trial / "infra-validation").mkdir(parents=True)
        (trial / "infra-validation" / midrun.SANDBOX_RECORD).write_text(json.dumps(
            {"sandbox_id": "sbx-1", "session_id": "task__abc__env", "environment_name": "e", "snapshot_id": "cut",
             "job": "j1"}))
        self.records = self.root / "runs" / "j1" / "midrun-snapshots.jsonl"

    def snapshotter(self, sandbox):
        return midrun.MidrunSnapshotter(self.job, record_path=self.records, every_s=600, poll_s=1,
                                        connect=lambda sid: sandbox)

    def test_periodic_start_and_compaction_triggers(self):
        ts20 = ["2026-01-01T00:00:00.000Z"] * 20
        sbx = FakeSandbox([
            # one tick = probe (state), then a snapshot = probe before + probe after
            probe(20, 0, ts=ts20, ticks=ticks(T0 + 1.0)),
            probe(20, 0, ts=ts20, ticks=ticks(T0 + 1.0)),
            probe(20, 0, ts=ts20, ticks=ticks(T0 + 1.05, T0 + 2.0)),
            probe(20, 1, ts=ts20, ticks=ticks(T0 + 10.0)),   # the agent compacted between polls
            probe(20, 1, ts=ts20, ticks=ticks(T0 + 10.0)),
            probe(20, 1, ts=ts20, ticks=ticks(T0 + 10.9, T0 + 11.7)),
            probe(20, 1, ts=ts20, ticks=ticks(T0 + 20.0)),
        ])
        snap = self.snapshotter(sbx)
        self.assertEqual(snap.tick(), 1)
        self.assertEqual(snap.tick(), 1)          # compaction: compactions went 0 -> 1
        self.assertEqual(snap.tick(), 0)          # nothing due (not 600 s, no new compaction)
        recs = midrun.read_records(self.records)
        self.assertEqual([r["trigger"] for r in recs], ["start", "compaction"])
        self.assertEqual(recs[0]["snapshot_id"], "snap-1")
        self.assertEqual(recs[0]["trial"], "task__abc")
        for key in ("ts", "t", "session_line", "session_line_before", "session_line_after", "n_compactions",
                    "pause_s", "create_s", "sandbox_id"):
            self.assertIn(key, recs[0])

    def test_periodic_trigger_after_interval(self):
        ts = ["2026-01-01T00:00:00.000Z"] * 10
        sbx = FakeSandbox([probe(10, ts=ts, ticks=ticks(T0 + 1.0)), probe(10, ts=ts, ticks=ticks(T0 + 1.0)),
                           probe(10, ts=ts, ticks=ticks(T0 + 1.05, T0 + 2.0)),
                           probe(10, ts=ts, ticks=ticks(T0 + 2.0)), probe(10, ts=ts, ticks=ticks(T0 + 2.0)),
                           probe(10, ts=ts, ticks=ticks(T0 + 2.05, T0 + 3.0))])
        snap = self.snapshotter(sbx)
        snap.tick()
        snap.state["sbx-1"]["last_t"] -= 601
        snap.tick()
        self.assertEqual([r["trigger"] for r in midrun.read_records(self.records)], ["start", "periodic"])

    def test_failed_snapshot_is_recorded_and_does_not_raise(self):
        class Boom(FakeSandbox):
            def create_snapshot(self, request_timeout=300):
                raise RuntimeError("HTTP 500")

        sbx = Boom([probe(10, ticks=ticks(T0 + 1.0)), probe(10, ticks=ticks(T0 + 1.1))])
        snap = self.snapshotter(sbx)
        self.assertEqual(snap.tick(), 0)
        self.assertTrue(snap.errors and "HTTP 500" in snap.errors[0])
        self.assertEqual(midrun.read_records(self.records), [])

    def test_sandbox_receipts(self):
        recs = midrun.sandbox_records(self.job)
        self.assertEqual([r["sandbox_id"] for r in recs], ["sbx-1"])
        self.assertEqual(recs[0]["trial"], "task__abc")
