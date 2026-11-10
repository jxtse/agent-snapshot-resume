"""Agent adapters. See base.py for the interface and docs/adding-an-agent.md for a walkthrough.

Registered adapters (name -> module:class):
"""
from __future__ import annotations

import importlib

ADAPTERS = {
    "claude-code": "agent_snapshot_resume.agents.claude_code:ClaudeCodeAdapter",
}


def get_adapter(name: str):
    """Return an adapter instance by registry name."""
    try:
        target = ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(ADAPTERS)}") from None
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)()
