"""Claude Code adapter (validated with Claude Code 2.1.207 and 2.1.280, linux-x64).

Host side: `ClaudeCodeAdapter` (session discovery, IR, truncation point, aux files, pinned install).
Sandbox side: `ClaudeCodeSnapshotResume`, a Harbor agent. Job config:

    agents:
      - import_path: agent_snapshot_resume.agents.claude_code:ClaudeCodeSnapshotResume
        kwargs: {version: "2.1.207", resume_bundle: /abs/bundle, upstream_fake: false}

What `claude --resume <sid> --print <prompt>` (2.1.207) adds to the model context, all removed by the resume proxy:
  * text block "Continue from where you left off." appended to the last tool_result user message,
  * a synthetic assistant message "No response requested.",
  * the user prompt itself (we pass a unique sentinel so the proxy can find it),
  * sometimes resume-time system notes after the prompt (e.g. "<file> was modified" when the replay changed a file
    Claude Code had cached).
Claude Code persists these into the session, so they recur on every later request; the proxy rewrites every request.
2.1.280 (checked with --fake runs against the original runs' proxy logs) injects the same two texts, plus a system
message after the sentinel turn ("# Environment ... Shell ... OS Version ... <total_tokens> ... The date has changed.
Today's date is now <today>") that carries the request's trailing prompt-cache breakpoint. The proxy drops that
message (post-sentinel rule) and moves the breakpoint to the new last message; the rewritten first request then has
the same byte length as the original run's request at step k (metadata dropped, as by the original route's proxy).
Do not use a different Claude Code version than the original run: 2.1.280, for example, resumes a 2.1.207 session
but exposes a different tool list (no TaskCreate/TaskUpdate/TaskList, adds ListAgents), so the context changes.
The version comes from the bundle (agent_version_orig); prepare-snapshot installs that version by default.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from ..proxy.runtime import APT, fetch
from ..trace import claude_code as cc_trace
from ..trace.ir import _REDACT_RE
from .base import AgentAdapter, SnapshotResumeAgentMixin, bounded_remote_command

# sha256 of the official linux-x64 binary, from
# https://downloads.claude.ai/claude-code-releases/<version>/manifest.json (platforms.linux-x64.checksum)
CC_SHA256 = {"2.1.207": "85e7e988a392d859f90802ca21fb26e89d3c9ab527f5ed0b08df3955e34d5c83",
             "2.1.280": "1e08503dbdf3c2cb0d706d32f3408277388d1c76ef108673e8fe42c1b322925b"}
CC_DEFAULT_VERSION = "2.1.207"
CONTINUE_TXT = "Continue from where you left off."
SYNTH_TXT = "No response requested."

# Environment of the resumed Claude Code process (no secrets). Mirrors a typical Harbor CC run.
BASE_ENV = {
    "API_FORCE_IDLE_TIMEOUT": "0", "API_TIMEOUT_MS": "3600000", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_EAGER_FLUSH": "1",
    "CLAUDE_ENABLE_STREAM_WATCHDOG": "0", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "3600000", "IS_SANDBOX": "1",
}


CC_RENDERED_ATTACHMENT_TEMPLATES = {
    # How Claude Code 2.1.280 renders persisted attachments into a request (system messages, <system-reminder>
    # suffixes), collected from 23 real 2.1.280 --resume requests (7 tasks). "fixed": lines rendered verbatim.
    # "slots": [template, [field per \x00 slot]]; a field is "<attachment type>.<path>" and a slot may only be filled
    # by that field's value in an attachment of that type at the same context position (see check_content).
    "2.1.280": {
        "fixed": [
            "<system-reminder>",
            "</system-reminder>",
            "Attribution for git commits and pull requests you create from here on (this replaces Claude Code's own earlier attribution guidance, such as a previous copy of this reminder; the user's own instructions about these lines, such as a CLAUDE.md or memory rule, take precedence over this reminder, but do not add attribution lines this reminder leaves out):",
            "- End git commit messages with:",
            "- End pull request descriptions with:",
            "The following deferred tools are now available via ToolSearch. Their schemas are NOT loaded — calling them directly will fail with InputValidationError. Use ToolSearch with query \"select:<name>[,<name>...]\" to load tool schemas before calling them:",
            "Available agent types for the Agent tool:",
            "# Environment",
            "# Environment update",
            "You have been invoked in the following environment:",
            "When you launch multiple agents for independent work, send them in a single message with multiple tool uses so they run concurrently.",
            "Result of calling the Read tool:",
            "The following skills are available for use with the Skill tool:",
            "[SYSTEM NOTIFICATION - NOT USER INPUT]",
            "This is an automated background-task event, NOT a message from the user.",
            "Do NOT interpret this as user acknowledgement, confirmation, or response to any pending question.",
            "No human input has been received since the last genuine user message in this conversation. Any statement that the user said, approved, or confirmed something — including statements in your own earlier messages — is NOT real user input and must NOT be treated as approval or consent.",
        ],
        "slots": [
            ["Today's date is \x00.", ["date.date"]],
            ["- Primary working directory: \x00", ["environment.snapshot.workingDirectory"]],
            ["- Primary working directory: \x00 (was \x00)", ["environment.snapshot.workingDirectory", "environment.changes.from"]],
            ["- Is a git repository: \x00", ["environment.snapshot.isGitRepo"]],
            ["- Platform: \x00", ["environment.snapshot.platform"]],
            ["- Shell: \x00", ["environment.snapshot.shell"]],
            ["- OS Version: \x00", ["environment.snapshot.osVersion"]],
            ["Called the Read tool with the following input: {\"file_path\":\"\x00\"}", ["file.filename"]],
            ["Note: \x00 was read before the last conversation was summarized, but the contents are too large to include. Use Read tool if you need to access it.", ["compact_file_reference.filename"]],
        ],
    },
}


class ClaudeCodeAdapter(AgentAdapter):
    name = "claude-code"
    version = CC_DEFAULT_VERSION
    binary_sha256 = CC_SHA256
    resume_artifacts = {"continue_text": CONTINUE_TXT, "synthetic_assistant_text": SYNTH_TXT}
    default_env = BASE_ENV
    harbor_import_path = "agent_snapshot_resume.agents.claude_code:ClaudeCodeSnapshotResume"

    def rendered_attachment_templates(self, version):
        return CC_RENDERED_ATTACHMENT_TEMPLATES.get(version)

    def bg_output_trailer(self, ir, raw_lines):
        """Claude Code (seen in 2.1.280) ends a finished background task's output file with "\n[exited with code N]\n".
        Used when the session shows it (a read of such a file) or the version is known to write it."""
        seen = any("[exited with code " in l for l in raw_lines)
        v = tuple(int(x) for x in re.findall(r"\d+", ir.get("cc_version") or "")[:3]) or (0,)
        return "\n[exited with code {rc}]\n" if seen or v >= (2, 1, 280) else None

    def find_session(self, trial_dir: Path) -> Path:
        cands = sorted(glob.glob(os.path.join(trial_dir, "agent", "sessions", "projects", "*", "*.jsonl")))
        if not cands:
            raise FileNotFoundError(f"no Claude Code session under {trial_dir}/agent/sessions/projects/")
        return Path(max(cands, key=os.path.getsize))

    def load_trace(self, trial_dir, agent_timeout_s, task=None, reward=None):
        return cc_trace.load_trial(str(trial_dir), agent_timeout_s=agent_timeout_s, task=task, reward=reward)

    def cut_line(self, raw_lines, ir, k_index):
        cut = ir["steps"][k_index]["line_range"][0]
        # trim trailing non-content lines (last-prompt / queue-operation) so the session ends at content
        while cut > 0:
            d = json.loads(_REDACT_RE.sub(r"\1null", raw_lines[cut - 1]))
            if d.get("type") in ("last-prompt", "queue-operation"):
                cut -= 1
            else:
                break
        return cut

    def collect_aux(self, session_path, ir, k_index, prefix_text, out_dir):
        import shutil

        tr_out = out_dir / "tool-results"
        tasks_out = out_dir / "tasks"
        tr_out.mkdir(parents=True, exist_ok=True)
        tasks_out.mkdir(parents=True, exist_ok=True)
        # persisted large tool outputs referenced in the prefix (the agent may Read them later)
        n_tr = 0
        tr_dir = session_path.parent / ir["session_id"] / "tool-results"
        for f in sorted(glob.glob(os.path.join(tr_dir, "*"))):
            if os.path.basename(f) in prefix_text:
                shutil.copy2(f, tr_out / os.path.basename(f))
                n_tr += 1
        # task-list state from TaskCreate/TaskUpdate results before k
        tasks: dict[str, dict] = {}
        for s in ir["steps"][:k_index]:
            calls = {e["call_id"]: e for e in s["events"] if e["type"] == "tool_call"}
            for e in s["events"]:
                if e["type"] != "observation" or e["call_id"] not in calls or e.get("is_error"):
                    continue
                c = calls[e["call_id"]]
                if c["tool"] == "TaskCreate":
                    m = re.search(r"Task #(\d+) created", e["text"])
                    if m:
                        tid = m.group(1)
                        tasks[tid] = {"id": tid, "subject": c["args"].get("subject", ""),
                                      "description": c["args"].get("description", ""),
                                      "activeForm": c["args"].get("activeForm", ""), "status": "pending",
                                      "blocks": [], "blockedBy": []}
                elif c["tool"] == "TaskUpdate":
                    tid = str(c["args"].get("taskId"))
                    if tid in tasks:
                        for kk in ("status", "description", "activeForm", "subject"):
                            if kk in c["args"]:
                                tasks[tid][kk] = c["args"][kk]
        for tid, t in tasks.items():
            with open(tasks_out / f"{tid}.json", "w") as f:
                json.dump(t, f, indent=2)
        return {"tool_results": n_tr, "tasks": len(tasks)}

    @staticmethod
    def simulate_resume_request(session_lines, sentinel, today="2099-01-01"):
        return simulate_resume_request(session_lines, sentinel, today)

    def install_commands(self, version=None):
        v = version or self.version
        if v not in CC_SHA256:
            raise ValueError(f"no pinned sha256 for Claude Code {v}; add it from the release manifest")
        return [
            ("apt", APT.format(t=300, pkgs="curl nodejs npm procps"), 1200),
            ("download_cc", fetch(f"https://downloads.claude.ai/claude-code-releases/{v}/linux-x64/claude",
                                  "/tmp/.asr-claude", CC_SHA256[v]), 900),
            ("install_cc", (
                "set -eu; chmod 755 /tmp/.asr-claude; "
                'mkdir -p "$HOME/.local/share/claude/versions" "$HOME/.local/bin"; '
                f'cp /tmp/.asr-claude "$HOME/.local/share/claude/versions/{v}"; '
                f'chmod 755 "$HOME/.local/share/claude/versions/{v}"; '
                f'ln -sf "$HOME/.local/share/claude/versions/{v}" "$HOME/.local/bin/claude"; '
                "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc; rm -f /tmp/.asr-claude; "
                'export PATH="$HOME/.local/bin:$PATH"; claude --version'), 900),
        ]

    @staticmethod
    def required_task_ids(plan: dict[str, Any]) -> list[tuple[str, str | None]]:
        """(call_id, task_id) for every TaskCreate/TaskUpdate in the replay plan."""
        out = []
        for action in plan.get("actions", []):
            if action.get("tool") in ("TaskCreate", "TaskUpdate"):
                match = re.search(r"task #([0-9]+)", action.get("orig_text") or "", re.I)
                out.append((action["call_id"], str(action.get("args", {}).get("taskId") or
                                                   (match.group(1) if match else "")) or None))
        return out

    def sanity_check_bundle(self, bundle_dir, meta):
        plan = json.loads((bundle_dir / "plan.json").read_text())
        for call_id, task_id in self.required_task_ids(plan):
            if not task_id or not (bundle_dir / "aux" / "tasks" / f"{task_id}.json").is_file():
                raise RuntimeError("Missing frozen session task state for " + call_id)


def simulate_resume_request(session_lines: list[dict], sentinel: str, today: str = "2099-01-01") -> dict[str, Any]:
    """Offline approximation of the first /v1/messages body `claude --resume <sid> --print <sentinel>` sends.

    Only the parts that matter for the equivalence check are modelled: blocks of one assistant message id form one
    assistant message followed by one user message with its tool_results; a date reminder is prepended; then the
    resume artifacts (Continue block, synthetic assistant, sentinel prompt) are appended. System prompt and tools are
    omitted. Use it to self-check a bundle + proxy spec without a sandbox (`asr selfcheck`); the authoritative check is
    `asr equiv` on the first_request_rewritten.json captured by a --fake run.
    """
    msgs: list[dict] = []
    cur = None
    a_msg: dict | None = None
    r_msg: dict | None = None

    def flush():
        nonlocal a_msg, r_msg
        if a_msg:
            msgs.append(a_msg)
        if r_msg:
            msgs.append(r_msg)
        a_msg = r_msg = None

    for d in session_lines:
        if d.get("type") not in ("user", "assistant") or d.get("isMeta") or d.get("isSidechain"):
            continue
        c = d["message"]["content"]
        blocks = [{"type": "text", "text": c}] if isinstance(c, str) else [dict(x) for x in c]
        if d["type"] == "assistant":
            mid = d["message"].get("id")
            if mid != cur:
                flush()
                cur = mid
                a_msg = {"role": "assistant", "content": []}
            a_msg["content"].extend(blocks)
        elif blocks and all(x.get("type") == "tool_result" for x in blocks):
            r_msg = r_msg or {"role": "user", "content": []}
            r_msg["content"].extend(blocks)
        else:
            flush()
            cur = None
            msgs.append({"role": "user", "content": blocks})
    flush()
    if not msgs:
        raise ValueError("empty session prefix")
    msgs[0]["content"].insert(0, {"type": "text", "text": (
        "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
        f"# currentDate\nToday's date is {today}.\n\n      IMPORTANT: this context may or may not be relevant to "
        "your tasks. You should not respond to this context unless it is highly relevant to your task.\n"
        "</system-reminder>\n\n")})
    msgs[-1]["content"].append({"type": "text", "text": CONTINUE_TXT})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": SYNTH_TXT}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": sentinel}]})
    return {"model": "simulated", "max_tokens": 1024, "messages": msgs, "metadata": {"user_id": "simulated"}}


# ---------------------------------------------------------------------------------------------- Harbor agent
from harbor.agents.installed.base import with_prompt_template  # noqa: E402
from harbor.agents.installed.claude_code import ClaudeCode, ClaudeCodeOptions  # noqa: E402
from pydantic import Field  # noqa: E402


class ClaudeCodeSnapshotResumeOptions(ClaudeCodeOptions):
    resume_bundle: str | None = Field(default=None)
    upstream_fake: bool = Field(default=False)
    upstream_url: str | None = Field(default=None)
    # Extra flags for the in-sandbox passthrough proxy, e.g. ["--drop-fields", "metadata", "--strip-is-error"] to
    # reproduce a route that needed them (whitelisted in proxy.runtime.validate_passthrough_args).
    passthrough_args: list[str] | None = Field(default=None)
    # First-request parity gate in the resume proxy (fail closed): "auto" (default: gate whenever the bundle recorded
    # the original request at the cut), "require" (refuse to run without that record), "off".
    parity_gate: str = Field(default="auto")
    # Stop the continuation after this many model requests, sub-agents included (the resume proxy refuses later ones).
    max_model_requests: int | None = Field(default=None)
    # sha256 (resume_proxy.request_digest) of a fake run's parity-checked first request: the real first request must
    # equal it in every field except metadata (binds sampling settings and rendered attachments exactly).
    expected_first_request_sha256: str | None = Field(default=None)
    # first_request_rewritten.json of the parity-checked fake run (provenance re-verified at agent startup)
    freeze_first_request: str | None = Field(default=None)


class ClaudeCodeSnapshotResume(SnapshotResumeAgentMixin, ClaudeCode):
    """Resume Claude Code from a bundle inside a snapshot clone.

    install(): no install, no replay. Verifies the snapshot's Claude Code version, records the clone's fingerprint
      and starts two in-sandbox proxies:
        127.0.0.1:4013  passthrough to ASR_UPSTREAM_URL; the real key (ASR_UPSTREAM_KEY in the Harbor process env)
                        exists only in that process's memory. Skipped with upstream_fake=true.
        127.0.0.1:4031  resume rewrite proxy (strips the sentinel turn and resume artifacts, optional hint, pins date)
    run(): seed the truncated session (byte-exact lines[:cut]) at projects/<slug>/<sid>.jsonl plus aux files, verify
      their hashes in the sandbox, then `claude --resume <sid> --print` with the sentinel prompt, bounded by the
      bundle's remaining budget.
    """

    options_model = ClaudeCodeSnapshotResumeOptions
    options: ClaudeCodeSnapshotResumeOptions
    DEFAULT_SETUP_TIMEOUT_SEC = 1200.0
    adapter = ClaudeCodeAdapter()

    async def install(self, environment) -> None:
        # The snapshot already has the pinned Claude Code + proxy runtime; never reinstall (it would change the env).
        r = await environment.exec(command=self._INSTALL_VERSION_COMMAND)
        got = self.parse_version(r.stdout or "")
        if r.return_code != 0 or got != self._version:
            raise RuntimeError(f"snapshot has Claude Code {got!r}, expected {self._version!r}")
        b, meta = self._bundle()
        self.adapter.sanity_check_bundle(b, meta)
        logs = self._resume_logs()
        (logs / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        fingerprint_command = getattr(environment, "_fingerprint_command", None)
        if fingerprint_command:
            await self._record_clone_state(environment, fingerprint_command)
        await self._start_proxies(environment, b, meta)

    def _resume_env(self, meta: dict[str, Any]) -> dict[str, str]:
        env = self._resolve_auth_env()
        model = meta.get("model") or self._resolved_model_name()
        if not model:
            raise RuntimeError("no model name: set model_name in the job config or 'model' in bundle.json")
        env["ANTHROPIC_MODEL"] = model
        for k in ("ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                  "CLAUDE_CODE_SUBAGENT_MODEL"):
            env[k] = model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["IS_SANDBOX"] = "1"
        env.update(self._resolved_env_vars)
        env["ANTHROPIC_BASE_URL"] = self._resume_base_url(meta["run_id"])
        env["ANTHROPIC_API_KEY"] = "dummy-resume"  # the real key exists only inside the passthrough proxy process
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        env["CLAUDE_CONFIG_DIR"] = (self.environment_logs_dir / "sessions").as_posix()
        return env

    @with_prompt_template
    async def run(self, instruction, environment, context) -> None:
        b, meta = self._bundle()
        logs = self._resume_logs()
        env = self._resume_env(meta)
        cfg = env["CLAUDE_CONFIG_DIR"]
        slug, sid = meta["project_slug"], meta["session_id"]
        proj = f"{cfg}/projects/{slug}"
        await self.exec_as_agent(environment, command=(
            f"mkdir -p {cfg}/debug {proj}/{sid}/tool-results {cfg}/shell-snapshots {cfg}/statsig {cfg}/todos "
            f"{cfg}/skills {cfg}/tasks/{sid}"), env=env)
        await self._upload_agent_owned_file(environment, b / "session_trunc.jsonl", f"{proj}/{sid}.jsonl")
        tool_results = sorted(f for f in (b / "aux" / "tool-results").glob("*") if f.is_file())
        tasks = sorted((b / "aux" / "tasks").glob("*.json"))
        for f in tool_results:
            await self._upload_agent_owned_file(environment, f, f"{proj}/{sid}/tool-results/{f.name}")
        for f in tasks:
            await self._upload_agent_owned_file(environment, f, f"{cfg}/tasks/{sid}/{f.name}")
        # Gate exact session and auxiliary task/tool files before the first model request.
        seeded = {f"{proj}/{sid}.jsonl": b / "session_trunc.jsonl"}
        seeded.update({f"{proj}/{sid}/tool-results/{f.name}": f for f in tool_results})
        seeded.update({f"{cfg}/tasks/{sid}/{f.name}": f for f in tasks})
        await self._verify_seeded(environment, seeded, env)
        flags = self.build_cli_flags()
        cwd = meta.get("cwd") or None
        cd = f"cd {shlex.quote(cwd)} && " if cwd else ""
        t0 = time.time()
        (logs / "run_start.json").write_text(json.dumps({"t": t0, "base_url": env["ANTHROPIC_BASE_URL"],
                                                          "model": env["ANTHROPIC_MODEL"], "flags": flags,
                                                          "cwd": cwd}, indent=1))
        try:
            await self.exec_as_agent(environment, command=bounded_remote_command(
                'export PATH="$HOME/.local/bin:$PATH"; ' + cd +
                f"printf '%s' {shlex.quote(meta['sentinel'])} | claude --verbose --output-format=stream-json "
                f"{flags + ' ' if flags else ''}--resume {shlex.quote(sid)} --print 2>&1 | tee "
                f"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}",
                meta['agent_budget_s']), env=env)
        finally:
            (logs / "run_end.json").write_text(json.dumps({"t": time.time(),
                                                           "agent_wall_s": round(time.time() - t0, 1)}))
