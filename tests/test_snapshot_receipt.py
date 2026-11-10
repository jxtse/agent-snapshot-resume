"""Snapshot creation whose HTTP response is lost must recover the receipt, never dispatch a second snapshot."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent_snapshot_resume.sandbox import source


def _sbx(create_error=None, listed=()):
    sbx = MagicMock()
    if create_error:
        sbx.create_snapshot.side_effect = create_error
    else:
        sbx.create_snapshot.return_value = SimpleNamespace(snapshot_id="snap-ok", names=[])
    sbx.list_snapshots.return_value.next_items.return_value = [SimpleNamespace(snapshot_id=s, names=[]) for s in listed]
    sbx.commands.run.return_value = SimpleNamespace(stdout="files=1\nabc\n")
    return sbx


def _run_snapshot(sbx):
    rec, steps = {}, []
    run = lambda s, cmd, timeout=900: s.commands.run(cmd, timeout=timeout)  # noqa: E731
    step = lambda name, t0, **kw: steps.append((name, kw))  # noqa: E731
    source._snapshot_and_kill(sbx, rec, step, run, "true")
    return rec, steps


def test_normal_path():
    rec, _ = _run_snapshot(_sbx())
    assert rec["snapshot_id"] == "snap-ok" and "snapshot_receipt_recovered" not in rec


def test_lost_response_recovers_single_snapshot_without_redispatch(monkeypatch):
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    sbx = _sbx(httpx.RemoteProtocolError("stream reset"), listed=["snap-recovered"])
    rec, _ = _run_snapshot(sbx)
    assert rec["snapshot_id"] == "snap-recovered" and rec["snapshot_receipt_recovered"]
    assert sbx.create_snapshot.call_count == 1


def test_ambiguous_snapshots_fail_closed(monkeypatch):
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="2 snapshots"):
        _run_snapshot(_sbx(httpx.RemoteProtocolError("reset"), listed=["a", "b"]))


def test_no_snapshot_found_reraises(monkeypatch):
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    with pytest.raises(httpx.RemoteProtocolError):
        _run_snapshot(_sbx(httpx.RemoteProtocolError("reset"), listed=[]))


def test_kill_waits_while_snapshotting(monkeypatch):
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    sbx = MagicMock()
    sbx.kill.side_effect = [Exception("409: not allowed while SNAPSHOTTING"), None]
    source.kill_source(sbx)
    assert sbx.kill.call_count == 2
    other = MagicMock()
    other.kill.side_effect = Exception("500 boom")
    with pytest.raises(Exception, match="500"):
        source.kill_source(other)
