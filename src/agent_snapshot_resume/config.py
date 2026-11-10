"""Runtime configuration from environment variables (optionally loaded from a local .env file).

Nothing here is ever written into a Harbor job config or a bundle. See README "Configuration".
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_OWNER_TAG = "asr"

# Variables `load_env_file` may import from .env. ASR_UPSTREAM_KEY is included so a user can keep it in .env;
# it is only ever read from the Harbor *process* environment and passed into the sandbox over a pipe.
_ENV_PREFIXES = ("E2B_", "ASR_", "FC_SANDBOX_", "REGISTRY_")


def load_env_file(path: str | os.PathLike | None = None, override: bool = False) -> list[str]:
    """Load E2B_*/ASR_*/FC_SANDBOX_*/REGISTRY_* variables from a .env file into os.environ.

    Returns the names that were set (never the values). Missing file -> no-op.
    """
    from dotenv import dotenv_values

    p = Path(path) if path else Path.cwd() / ".env"
    if not p.is_file():
        return []
    loaded = []
    for key, value in dotenv_values(p).items():
        if value and key.startswith(_ENV_PREFIXES) and (override or not os.environ.get(key)):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def owner_tag() -> str:
    """Sandbox metadata `owner` and template-name prefix. Lowercase letters, digits and '-' only."""
    tag = os.environ.get("ASR_OWNER_TAG") or DEFAULT_OWNER_TAG
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", tag):
        raise ValueError("ASR_OWNER_TAG must match [a-z0-9][a-z0-9-]{0,40}")
    return tag


def upstream_url() -> str | None:
    """Anthropic-messages-compatible endpoint the in-sandbox passthrough proxy forwards to."""
    return os.environ.get("ASR_UPSTREAM_URL") or None


UPSTREAM_KEY_ENV = "ASR_UPSTREAM_KEY"


def upstream_key() -> str | None:
    """Model API key. Read only from the process environment; never logged, never written to disk."""
    return os.environ.get(UPSTREAM_KEY_ENV) or None


def model_override() -> str | None:
    return os.environ.get("ASR_MODEL") or None
