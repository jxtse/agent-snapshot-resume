"""The agent adapter interface.

Everything agent-specific lives in ONE module under agents/. An adapter has two halves:

1. `AgentAdapter` (host side, no Harbor import needed): used by `asr build-bundle` and `asr prepare-snapshot`.
     name                     registry name, e.g. "claude-code"
     version / binary_sha256  pinned agent release + sha256 of the linux-x64 artifact (resume must use the SAME
                              version as the original run: tool list and system prompt must not change)
     resume_artifacts         texts the agent injects when it resumes a session; the resume proxy removes them
     find_session(trial_dir)  locate the native session file inside a Harbor trial dir
     load_trace(...)          native session -> trajectory IR (shape in docs/design.md; consumed by trace/ir.py)
     cut_line(...)            index of the first session line that belongs to step k (byte-exact truncation point)
     collect_aux(...)         extra files the agent needs next to the session (persisted outputs, task lists, ...)
     install_commands(...)    shell steps that install the pinned agent in the source sandbox (checks sha256)
     sanity_check_bundle(...) adapter-specific consistency checks run before seeding (optional)

2. A Harbor agent class (sandbox side) that subclasses Harbor's installed agent and `SnapshotResumeAgentMixin`:
     install()  must NOT install anything: verify the snapshot's agent version, start the proxies
     run()      seed the truncated session + aux files, verify their hashes in the sandbox, then run the agent's own
                resume command with the bundle's sentinel as the prompt, bounded by `bounded_remote_command`
   The agent must be pointed at the resume proxy (base URL http://127.0.0.1:4031/r/<run_id>) with a DUMMY key.

The generic replayer (replay/replay_inside.py) understands Write / Edit / Bash / Agent / Task* / web actions and
Claude Code's observation conventions. An adapter for an agent with different tools maps them in load_trace()
(see docs/adding-an-agent.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AgentAdapter(ABC):
    name: str = ""
    version: str = ""
    binary_sha256: dict[str, str] = {}
    # {"continue_text": ..., "synthetic_assistant_text": ...}; written into proxy_spec.json
    resume_artifacts: dict[str, str] = {}

    @abstractmethod
    def find_session(self, trial_dir: Path) -> Path:
        """Path of the agent's native session file in a Harbor trial dir."""

    @abstractmethod
    def load_trace(self, trial_dir: Path, agent_timeout_s: float, task: str | None = None,
                   reward: float | None = None) -> dict[str, Any]:
        """Parse the native session into the trajectory IR."""

    @abstractmethod
    def cut_line(self, raw_lines: list[str], ir: dict[str, Any], k_index: int) -> int:
        """Number of leading session lines that form the resumed prefix (everything before step k)."""

    def collect_aux(self, session_path: Path, ir: dict[str, Any], k_index: int, prefix_text: str,
                    out_dir: Path) -> dict[str, int]:
        """Copy auxiliary files the resumed agent needs into out_dir (bundle/aux). Returns counts per kind."""
        return {}

    def bg_output_trailer(self, ir: dict[str, Any], raw_lines: list[str]) -> str | None:
        """Text the agent appends to a finished background task's output file ("{rc}" = exit code), or None."""
        return None

    @abstractmethod
    def install_commands(self, version: str | None = None) -> list[tuple[str, str, int]]:
        """[(step_name, shell_command, timeout_s)] run as root in the source sandbox. Must verify sha256."""

    def sanity_check_bundle(self, bundle_dir: Path, meta: dict[str, Any]) -> None:
        """Raise if the bundle cannot be seeded faithfully (optional)."""


def bounded_remote_command(command: str, budget_s: float) -> str:
    """Wrap a command so the sandbox itself enforces the remaining agent budget (even if the host controller dies).

    `timeout -s TERM -k 10s` kills the whole child, and `bash -o pipefail` keeps a failing agent visible through tee.
    """
    budget_s = float(budget_s)
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError('Remote continuation budget must be finite and positive')
    return (f'timeout -s TERM -k 10s {budget_s:.6f}s '
            f'bash -o pipefail -c {shlex.quote(command)}')


def load_bundle(path: str | os.PathLike) -> tuple[Path, dict[str, Any]]:
    b = Path(path or "")
    if not (b / "bundle.json").is_file():
        raise FileNotFoundError(f"resume_bundle missing bundle.json: {b}")
    return b, json.loads((b / "bundle.json").read_text())


def seed_hash_command(paths: list[str]) -> str:
    code = ("import pathlib,hashlib,json; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() "
            "for p in " + repr(list(paths)) + "}))")
    return "python3 -c " + shlex.quote(code)


class SnapshotResumeAgentMixin:
    """Shared sandbox-side logic for Harbor agents that resume from a snapshot. Mix in BEFORE the Harbor agent class.

    Expects `self.options.resume_bundle`, `self.options.upstream_fake`, `self.options.upstream_url`,
    `self.logs_dir`, `self.exec_as_root`, `self.exec_as_agent`.
    """

    RESUME_PORT = 4031
    PASSTHROUGH_PORT = 4013

    def _bundle(self):
        return load_bundle(self.options.resume_bundle)

    def _resume_logs(self) -> Path:
        d = self.logs_dir / "resume"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _resume_base_url(self, run_id: str) -> str:
        return f"http://127.0.0.1:{self.RESUME_PORT}/r/{run_id}"

    async def _record_clone_state(self, environment, fingerprint_command: str) -> None:
        logs = self._resume_logs()
        fp = await environment.exec(command=fingerprint_command, user="root", timeout_sec=600)
        ps = await environment.exec(command="ps -eo pid,etime,args --sort=pid | grep -v -E 'ps -eo|grep' | tail -20",
                                    user="root")
        (logs / "clone_env.json").write_text(json.dumps({"fingerprint": (fp.stdout or "").strip(),
                                                         "processes": (ps.stdout or "").strip()}, indent=1))

    async def _start_proxies(self, environment, bundle: Path, meta: dict[str, Any]) -> None:
        """Start the passthrough proxy (unless fake) and the resume proxy in the sandbox."""
        from .. import config
        from ..proxy import runtime

        logs = self._resume_logs()
        spec = json.loads((bundle / "proxy_spec.json").read_text())
        upstream = "fake" if self.options.upstream_fake else f"http://127.0.0.1:{self.PASSTHROUGH_PORT}"
        spec.update(upstream=upstream)
        spec.pop("key_file", None)
        if spec.get("model"):
            raise ValueError("proxy-side model override is not supported; set the agent model in the job config")
        gate = getattr(self.options, "parity_gate", "auto") or "auto"
        if gate != "off":
            from ..proxy.resume_proxy import route_from_passthrough_args
            exp = meta.get("orig_request_at_k")
            if exp:
                spec["first_request_parity"] = dict(exp, route=route_from_passthrough_args(
                    getattr(self.options, "passthrough_args", None) or []))
                if gate == "require":
                    if not exp.get("content"):
                        raise RuntimeError("parity_gate=require but the bundle has no content reference; rebuild it")
                    spec["first_request_parity"]["content"] = dict(exp["content"], require_complete=True)
            elif gate == "require":
                raise RuntimeError("parity_gate=require but the bundle has no orig_request_at_k (original proxy log "
                                   f"missing or ambiguous: {meta.get('orig_request_locate_error')}); "
                                   "use parity_gate=auto/off to run without the first-request parity gate")
        if getattr(self.options, "max_model_requests", None):
            spec["max_model_requests"] = int(self.options.max_model_requests)
        if getattr(self.options, "expected_first_request_sha256", None):
            spec["expected_first_request_sha256"] = self.options.expected_first_request_sha256
        (logs / "proxy_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        t0 = time.time()
        if not self.options.upstream_fake:
            url = self.options.upstream_url or config.upstream_url()
            if not url:
                raise RuntimeError("Set ASR_UPSTREAM_URL (or the agent kwarg upstream_url) to the model endpoint")
            key = config.upstream_key()
            if not key:
                raise RuntimeError(f"{config.UPSTREAM_KEY_ENV} is not set in the Harbor process environment")
            # The key travels only as the env of this single exec (redacted by Harbor's logger: name contains KEY)
            # and from there into the proxy over a pipe. See proxy/runtime.py.
            extra = runtime.validate_passthrough_args(getattr(self.options, "passthrough_args", None) or [])
            await self.exec_as_root(environment, command=runtime.passthrough_start_command(
                url, self.PASSTHROUGH_PORT, extra=" ".join(shlex.quote(x) for x in extra)),
                env={runtime.KEY_ENV_IN_SANDBOX: key})
        resume_proxy = Path(runtime.HERE / "resume_proxy.py")
        await self.exec_as_root(environment, command=f"mkdir -p {runtime.PROXY_DIR}/resume/specs")
        await environment.upload_file(resume_proxy, f"{runtime.PROXY_DIR}/resume/resume_proxy.py")
        await environment.upload_file(logs / "proxy_spec.json", f"{runtime.PROXY_DIR}/resume/specs/{meta['run_id']}.json")
        await self.exec_as_root(environment, command=runtime.resume_start_command(self.RESUME_PORT))
        (logs / "setup.json").write_text(json.dumps({
            "proxies_s": round(time.time() - t0, 1),
            "passthrough_args": getattr(self.options, "passthrough_args", None) or [],
            "resume_proxy_sha256": hashlib.sha256(resume_proxy.read_bytes()).hexdigest(),
            "upstream_fake": bool(self.options.upstream_fake)}, indent=1))

    async def _verify_seeded(self, environment, seeded: dict[str, Path], env: dict[str, str]) -> None:
        """Fail closed unless every seeded remote file has the exact local bytes."""
        expected = {remote: hashlib.sha256(local.read_bytes()).hexdigest() for remote, local in seeded.items()}
        result = await self.exec_as_agent(environment, command=seed_hash_command(list(seeded)), env=env)
        actual = json.loads(result.stdout or "{}")
        (self._resume_logs() / "session_aux_hashes.json").write_text(
            json.dumps({"expected": expected, "actual": actual}, indent=1))
        if actual != expected:
            raise RuntimeError("Session/auxiliary seed hash mismatch")
