"""Offline comparison of independently collected cut evidence. Never dispatches."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    """Each path names actual captured bytes in an independently acquired export."""

    def services(self) -> set[str]: ...
    def observation(self, service: str, capability: str) -> Path: ...


class DirectoryBackend:
    def __init__(self, root: Path):
        self.root = root / "services"

    def services(self):
        return {entry.name for entry in self.root.iterdir() if entry.is_dir()}

    def observation(self, service, capability):
        return self.root / service / capability


CAPABILITIES = ("checkpoint", "network", "image", "volumes", "processes", "binaries", "limits", "files", "output")
CONTEXT = ("session", "branch", "prefix", "children", "tools", "system", "effort", "deadline", "cut")


def compare_capture(reference: Path, observed: Path) -> bool:
    """Require distinct artifacts. Provenance of the two acquisitions is the caller's responsibility."""
    return (reference.is_file() and observed.is_file() and reference.stat().st_size > 0
            and not os.path.samefile(reference, observed)
            and reference.stat().st_size == observed.stat().st_size
            and equal_bytes(reference, observed))


def equal_bytes(left, right):
    with left.open("rb") as a, right.open("rb") as b:
        while chunk := a.read(1024 * 1024):
            if chunk != b.read(len(chunk)):
                return False
        return not b.read(1)


def environment(reference: Backend, restored: Backend):
    try:
        expected = reference.services()
        actual = restored.services()
    except FileNotFoundError:
        return ["services"]
    problems = [] if expected and expected == actual else ["services"]
    for service in sorted(expected & actual):
        for capability in CAPABILITIES:
            if not compare_capture(reference.observation(service, capability),
                                   restored.observation(service, capability)):
                problems.append(f"{service}/{capability}")
    return problems


def unique_object(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON field")
    return obj


def body_from(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object)


def request_matches(reference: Path, observed: Path, hint: dict | None):
    """Compare complete JSON bodies, allowing one explicitly positioned added user text block."""
    if not reference.is_file() or not observed.is_file() or reference.samefile(observed):
        return False
    try:
        original = body_from(reference)
        candidate = body_from(observed)
        if not isinstance(original, dict) or not isinstance(candidate, dict) or not original.get("model"):
            return False
        if ("input" in original) == ("messages" in original):
            return False
        responses = "input" in original
        history = "input" if responses else "messages"
        if not isinstance(original[history], list) or not original[history]:
            return False
        if hint is not None:
            mi, bi, text = hint["message"], hint["block"], hint["text"]
            if not (type(mi) is int and type(bi) is int and min(mi, bi) >= 0 and isinstance(text, str) and text):
                return False
            from .proxy.native_wire import user_item
            messages = candidate[history]
            message = messages[mi]
            if message.get("role") != "user":
                return False
            if not ("system" in original and not responses) and bi == 0 and message == user_item(
                    "responses" if responses else "chat", text):
                messages.pop(mi)
            else:
                block_type = "input_text" if responses else "text"
                if message["content"].pop(bi) != {"type": block_type, "text": text}:
                    return False
        return json.dumps(original, sort_keys=True, allow_nan=False) == json.dumps(candidate, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def compare_environment_exports(reference: Path, restored: Path):
    original = DirectoryBackend(reference)
    problems = environment(original, DirectoryBackend(restored))
    if not compare_capture(reference / "topology.json", restored / "topology.json"):
        return [*problems, "topology"]
    try:
        services = body_from(reference / "topology.json")["services"]
        if (not isinstance(services, list) or not services or not all(isinstance(s, str) and s for s in services)
                or len(services) != len(set(services)) or set(services) != original.services()):
            problems.append("topology")
    except (ValueError, TypeError, KeyError, OSError):
        problems.append("topology")
    return problems


def compare_exports(reference: Path, restored: Path, hint: dict | None = None, control: Path | None = None):
    env = compare_environment_exports(reference, restored)
    context = [key for key in CONTEXT if not compare_capture(reference / key, restored / key)]
    if not request_matches(reference / "request.json", restored / "request.json", hint):
        context.append("first_request")
    harness_equal = compare_capture(reference / "harness", restored / "harness")
    harness_paths = [root / "harness" for root in (reference, restored)]
    if any(not path.is_file() or not path.stat().st_size for path in harness_paths):
        context.append("harness")
    elif harness_paths[0].samefile(harness_paths[1]):
        context.append("shared_harness_evidence")
    elif not harness_equal:
        # A changed harness needs an unhinted observation from that same harness at this cut.
        if (control is None or not compare_capture(control / "harness", restored / "harness")
                or compare_environment_exports(reference, control)
                or any(not compare_capture(reference / key, control / key) for key in CONTEXT)
                or not request_matches(reference / "request.json", control / "request.json", None)):
            context.append("changed_harness_control")
    condition = "original-harness" if harness_equal else "alternate-harness-diagnostic"
    return {"environment_problems": env, "context_problems": context, "same_harness": harness_equal,
            "condition": condition, "historical_restore_verified": False,
            "comparison": "blocked" if env or context else "captures-match", "dispatch": False}
