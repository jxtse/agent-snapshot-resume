"""Pi session tree adapter for the openai-completions provider."""
import json
import math
import posixpath
import re
import shlex
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from harbor.agents.installed.pi import Pi, PiOptions

from ..replay.replay_inside import pi_log_path
from ..trace.ir import bash_is_read_only
from .native import NativeAdapter, NativeOptions, NativeResumeMixin


# Content pin over all published package files, independent of npm tar headers.
# Verified against the local 1.4.0 distribution. Dependencies are resolved by npm.
PI_TARBALL_SHA256 = "d155c16ef6fa00960fbf58b2911536e80b05f33700e0b14fbd08967ede0c7428"
PI_TREE_SHA256 = "082e3550440ac551596aafa76e75b11cb05ed2a308a31c6ebdcd5d41a2f1700f"
VERIFY_PACKAGE = r"""
import hashlib, sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    tree = hashlib.sha256()
    entries = sorted((m.name, m) for m in archive.getmembers() if not m.isdir())
    for name, member in entries:
        if not member.isfile() or not name.startswith("package/"):
            sys.exit("unexpected Pi package member")
        tree.update(name[8:].encode() + b"\0" + hashlib.sha256(archive.extractfile(member).read()).digest())
    if tree.hexdigest() != sys.argv[2]:
        sys.exit("Pi package SHA256 mismatch")
"""


REASONING_PROVIDER = "asr-reasoning"
NODE_VERSION = "22.23.4"
NODE_SHA256 = "b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406"


def pi_models(endpoint, model, provider):
    descriptor = {"id": model}
    if provider == REASONING_PROVIDER:
        descriptor.update(reasoning=True, thinkingLevelMap={"max": "max"},
                          compat={"thinkingFormat": "deepseek", "supportsReasoningEffort": True,
                                  "requiresReasoningContentOnAssistantMessages": True})
    return {"providers": {provider: {"api": "openai-completions", "baseUrl": endpoint,
                                     "apiKey": "dummy", "models": [descriptor]}}}


def pi_args(session, model, provider, prompt):
    thinking = ["--thinking", "max"] if provider == REASONING_PROVIDER else []
    return [*thinking, "--offline", "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files",
            "--no-prompt-templates", "--no-themes", "--no-approve", "--tools", "read,bash,edit,write",
            "--session", str(session), "--provider", provider, "--model", model, "--mode", "json", "--print", "--", prompt]


def active_branch(records):
    by_id = {}
    for i, row in enumerate(records[1:], 1):
        parent, entry = row["parentId"], row["id"]
        if entry in by_id or parent is not None and parent not in by_id:
            raise ValueError("invalid Pi parent or duplicate entry id")
        by_id[entry] = (i, row)
    chain = []
    leaf = records[-1]["id"] if len(records) > 1 else None
    while leaf is not None:
        i, row = by_id[leaf]
        chain.append((i, row))
        leaf = row["parentId"]
    return reversed(chain)


def resolve_pi_path(value, cwd, home=None):
    """Match resolveToCwd on the POSIX guest. Home must come from the recording."""
    value = re.sub(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", value).removeprefix("@")
    if value == "~" or value.startswith("~/"):
        if not home or not posixpath.isabs(home):
            raise ValueError("Pi home-relative path has no recorded home directory")
        value = home + value[1:]
    elif value.startswith("file://"):
        url = urlsplit(value.replace("\\", "/"))
        if url.netloc.lower() not in ("", "localhost") or re.search(r"%2f|%(?![\da-f]{2})", url.path, re.I):
            raise ValueError("invalid POSIX file URL")
        value = unquote(url.path or "/", errors="strict")
    return "/" + posixpath.normpath(posixpath.join(cwd, value)).lstrip("/")


def tool_action(name, args, cwd, home=None):
    if name == "bash":
        command = args["command"]
        mapped = {"command": command, "pi_bash": True}
        if "timeout" in args:
            seconds = args["timeout"]
            if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                    or not 0 < seconds * 1000 <= 2_147_483_647):
                raise ValueError("invalid Pi Bash timeout")
            mapped["timeout"] = 1000 * seconds
        return "Bash", mapped, (
            "read_only" if bash_is_read_only(command) else "exec")
    if name in {"read", "write", "edit"}:
        path = resolve_pi_path(args["path"], cwd, home)
        if name == "read":
            return "Read", dict(file_path=path), "read_only"
        if name == "write":
            return "Write", dict(file_path=path, content=args["content"]), "file_write"
        edits = args.get("edits") or [args]
        if len(edits) == 1:
            return "Edit", dict(file_path=path, match_original=True, old_string=edits[0]["oldText"], new_string=edits[0]["newText"]), "file_write"
        if all(isinstance(e, dict) and "oldText" in e and "newText" in e for e in edits):
            return "MultiEdit", {"file_path": path, "match_original": True, "edits": [
                {"old_string": e["oldText"], "new_string": e["newText"]} for e in edits]}, "file_write"
    return name, args, "other"


class PiAdapter(NativeAdapter):
    name, version, wire = "pi", "1.4.0", "chat"
    binary_sha256 = {"1.4.0": PI_TARBALL_SHA256}
    session_root = "agent/pi/sessions"
    session_glob = session_root + "/**/*.jsonl"
    harbor_import_path = "agent_snapshot_resume.agents.pi:PiSnapshotResume"

    def load_trace(self, trial_dir, agent_timeout_s, task=None, reward=None):
        trial = Path(trial_dir)
        path = self.find_session(trial)
        with path.open("rb") as stream:
            records = [json.loads(line) for line in stream]
        header = records[0]
        if header["type"] != "session" or header.get("version") != 3:
            raise ValueError("unsupported Pi session schema")
        home_file = trial / "agent/pi/home.txt"
        home = home_file.read_text().rstrip("\n") if home_file.is_file() else None
        messages = (r.get("message") or {} for r in records)
        response_owners = Counter(m["responseId"] for m in messages if m.get("role") == "assistant"
                                  and m.get("responseId"))
        steps, calls = [], {}
        model = provider = None
        branch = list(active_branch(records))
        included = {i for i, _ in branch}
        abandoned_effects = []
        for i, row in enumerate(records[1:], 1):
            if i in included:
                continue
            msg = row.get("message") or {}
            if msg.get("role") == "toolResult" and (
                    (msg.get("details") or {}).get("fullOutputPath")
                    or any(pi_log_path(b.get("text")) for b in msg["content"])):
                abandoned_effects.append(i)
            if msg.get("role") == "bashExecution" or (msg.get("role") == "assistant" and any(
                    b.get("type") == "toolCall" and tool_action(b["name"], b["arguments"], header["cwd"], home)[2] != "read_only"
                    for b in msg["content"])):
                abandoned_effects.append(i)
        for i, row in branch:
            typ = row["type"]
            if typ == "compaction" and steps:
                steps[-1]["events"].append(dict(type="compaction_boundary", line=i))
            if typ != "message":
                continue
            msg = row["message"]
            role = msg["role"]
            if role in ("assistant", "user", "system"):
                steps.append(dict(source="agent" if role == "assistant" else "user", timestamp=row["timestamp"],
                                  line_range=[i], cut_parent=row["parentId"], message_id=row["id"], events=[]))
            if role == "bashExecution":
                steps.append(dict(source="user", timestamp=row["timestamp"], line_range=[i],
                                  events=[dict(type="tool_call", tool="Bash", side_effect="exec",
                                               call_id=row["id"], cwd=header["cwd"], args={"command": msg["command"]},
                                               timestamp=row["timestamp"], line=i, is_error=bool(msg.get("exitCode"))),
                                          dict(type="observation", call_id=row["id"], text=msg.get("output", ""),
                                               line=i, is_error=bool(msg.get("exitCode")))]))
                continue
            events = steps[-1]["events"]
            if role == "toolResult":
                call = calls[msg["toolCallId"]]
                call["is_error"] = msg["isError"]
                call["duration_s"] = msg.get("durationMs", 0) / 1000
                text = "\n".join(b["text"] for b in msg["content"] if b["type"] == "text")
                if call["tool"] == "Bash":
                    log = pi_log_path(text)
                    recorded_log = (msg.get("details") or {}).get("fullOutputPath")
                    if recorded_log and recorded_log != log:
                        raise ValueError("unsupported or inconsistent Pi full output path")
                    if log:
                        call["args"]["pi_output_file"] = log
                        call["side_effect"] = "exec"
                    timeout = re.search(r"(?:^|\n\n)Command timed out after (\d+(?:\.\d+)?) seconds\s*\Z", text)
                    if call["is_error"] and timeout:
                        seconds = float(timeout[1])
                        if seconds <= 0 or seconds * 1000 != call["args"].get("timeout"):
                            raise ValueError("Pi timeout differs from recorded tool arguments")
                        call["args"]["pi_timeout_status"] = timeout[1]
                        call["side_effect"] = "exec"
                    status = re.search(r"(?:^|\n+)Command exited with code (\d+)\s*\Z", text)
                    if call["is_error"] and status:
                        text = f"Exit code {status[1]}\n" + text[:status.start()]
                    elif not call["is_error"] and text == "(no output)":
                        text = ""
                events.append(dict(type="observation", call_id=call["call_id"], is_error=call["is_error"],
                                   text=text, line=i))
            elif role == "assistant":
                steps[-1].update(native_provider=msg["provider"], native_model=msg["model"])
                if isinstance(msg.get("timestamp"), (int, float)):
                    steps[-1]["request_start_s"] = msg["timestamp"] / 1000
                rid = msg.get("responseId")
                if rid and response_owners[rid] == 1 and msg.get("stopReason") not in {"error", "aborted"}:
                    steps[-1]["native_response_id"] = rid
                if msg.get("api") != "openai-completions":
                    raise ValueError("Pi adapter requires openai-completions")
                model, provider = msg["model"], msg["provider"]
                for block in msg["content"]:
                    if block["type"] == "toolCall":
                        tool, args, effect = tool_action(block["name"], block["arguments"], header["cwd"], home)
                        event = dict(type="tool_call", tool=tool, args=args, side_effect=effect, cwd=header["cwd"],
                                     call_id=block["id"], timestamp=row["timestamp"], line=i, is_error=None)
                        calls[block["id"]] = event
                        events.append(event)
                    elif block["type"] in ("text", "thinking"):
                        events.append(dict(type="assistant_text" if block["type"] == "text" else "assistant_reasoning",
                                           text=block[block["type"]]))
            elif role == "user":
                events.append(dict(type="user_text", text=msg["content"]))
        version_file = trial / "agent/pi/version.txt"
        result_file = trial / "result.json"
        version = version_file.read_text().strip() if version_file.is_file() else None
        if version is None and result_file.is_file():
            version = (json.loads(result_file.read_text()).get("agent_info") or {}).get("version")
        if version != self.version:
            raise ValueError("record the pinned Pi CLI version in agent/pi/version.txt")
        from .native import align_atif_step_ids
        flags = align_atif_step_ids(
            trial, steps, [[e["call_id"] for e in s["events"] if e["type"] == "tool_call"] for s in steps],
            [s["source"] if s["source"] == "agent" else "user" for s in steps])
        return self.trace(path, trial, records, header["id"], header["cwd"], version, steps, agent_timeout_s,
                          flags=flags, task=task, reward=reward, model=model, provider=provider,
                          abandoned_effects=abandoned_effects)

    def native_reference(self, trial, ir, k, prefix):
        logdir = trial / "agent/model-proxy"
        if (logdir / "native-reference.jsonl").is_file():
            return super().native_reference(trial, ir, k, prefix)
        rid = ir["steps"][k].get("native_response_id")
        matches = []
        if rid:
            for log in sorted(logdir.glob("requests-*.jsonl")):
                with log.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if (row.get("response_id") == rid and row.get("status") == 200
                                and row.get("path") == "/v1/chat/completions"):
                            matches.append(row.get("native_reference") or {})
        if (len(matches) == 1 and matches[0].get("wire") == self.wire and matches[0].get("complete") is True
                and matches[0].get("raw_sha256")):
            return dict(content=matches[0], method="captured-response-id", index=k), None
        return None, "no unique complete receipt for this Pi response"

    def cut_line(self, raw_lines, ir, k_index):
        parent = ir["steps"][k_index]["cut_parent"]
        line = super().cut_line(raw_lines, ir, k_index)
        while line and json.loads(raw_lines[line-1])["id"] != parent:
            line -= 1
        if any(i < ir["steps"][k_index]["line_range"][0] for i in ir["abandoned_effects"]):
            raise ValueError("Pi branch switch has historical workspace side effects outside the active branch")
        if not line:
            raise ValueError("Pi assistant parent missing")
        return line

    def install_commands(self, version=None):
        if (version or self.version) != self.version:
            raise ValueError("unsupported Pi version")
        url = "https://registry.npmjs.org/@earendil-works/pi-coding-agent/-/pi-coding-agent-1.4.0.tgz"
        archive = "/tmp/asr-pi-1.4.0.tgz"
        # Tarball sha256 (matches npm dist.integrity sha512 for 1.4.0); checked with coreutils so the image needs
        # no python3. VERIFY_PACKAGE/PI_TREE_SHA256 remain for host-side fixture checks.
        from ..proxy.runtime import fetch as _fetch
        command = (f"{_fetch(url, archive, PI_TARBALL_SHA256)} && "
                   f"npm install --prefix /opt/asr-pi --ignore-scripts --no-audit --no-fund {archive} && "
                   "ln -sf /opt/asr-pi/node_modules/.bin/pi /usr/local/bin/pi && pi --version")
        from ..proxy.runtime import fetch
        node_file = f"node-v{NODE_VERSION}-linux-x64.tar.gz"
        node = (fetch(f"https://nodejs.org/dist/v{NODE_VERSION}/{node_file}", "/tmp/asr-node.tgz", NODE_SHA256)
                + " && mkdir -p /opt/asr-node && tar xzf /tmp/asr-node.tgz -C /opt/asr-node --strip-components=1"
                  " && ln -sf /opt/asr-node/bin/node /usr/local/bin/node"
                  " && ln -sf /opt/asr-node/bin/npm /usr/local/bin/npm"
                + f' && test "$(node --version)" = "v{NODE_VERSION}"')
        # Minimal task images may have neither curl nor python3; install curl with the image's own package manager.
        bootstrap = ("command -v curl >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || "
                     "(export DEBIAN_FRONTEND=noninteractive; for i in 1 2 3; do "
                     "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && exit 0; "
                     "sleep 5; done; exit 1)")
        return [("downloader", bootstrap, 600), ("node", node, 300),
                ("pi", command + f' && test "$(pi --version)" = "{self.version}"', 600)]

    def resume_setup(self, meta, base):
        home = "/logs/agent/pi"
        session = self.replay_session_path(meta)
        model, provider = meta["native_model"], meta["native_provider"]
        return ({"PI_CODING_AGENT_DIR": home + "/config", "PI_OFFLINE": "1"}, session,
                ["pi", *pi_args(session, model, provider, meta["sentinel"])], home + "/config/models.json",
                json.dumps(pi_models(base + "/v1", model, provider)))


class PiSnapshotResumeOptions(NativeOptions, PiOptions):
    pass


class PiSnapshotResume(NativeResumeMixin, Pi):
    options_model = PiSnapshotResumeOptions
    adapter = PiAdapter()


class PiBaseline(Pi):
    """Fresh single-session baseline. Harbor applies the task's agent timeout."""
    def get_version_command(self):
        return "pi --version"

    async def install(self, environment):
        import certifi
        from .. import config
        from ..proxy import runtime

        for _, cmd, seconds in PiAdapter().install_commands(self._version):
            await self.exec_as_root(environment, command=cmd, timeout_sec=seconds)
        await self.exec_as_root(environment, command="mkdir -p /tmp/.asr-proxy-src")
        files = runtime.proxy_file_bytes()
        for name in files:
            await environment.upload_file(runtime.HERE / name, f"/tmp/.asr-proxy-src/{name}")
        sums = self.logs_dir / "proxy-SHA256SUMS"
        sums.write_text(runtime.sha256sums(files))
        await environment.upload_file(sums, "/tmp/.asr-proxy-src/SHA256SUMS")
        await environment.upload_file(Path(certifi.where()), "/tmp/.asr-cacert.pem")
        await self.exec_as_root(environment, command=runtime.proxy_place_command(), timeout_sec=1200)
        await self.exec_as_root(environment, command=runtime.CA_LINK)
        url, key = config.upstream_url(), config.upstream_key()
        if not url or not key:
            raise ValueError("configure the baseline upstream in the Harbor process")
        await self.exec_as_root(environment, command=runtime.passthrough_start_command(
            url, extra="--record-request-digests"), env={runtime.KEY_ENV_IN_SANDBOX: key})

    async def run(self, instruction, environment, context):
        from ..proxy import runtime
        model = (self.model_name or "").removeprefix(REASONING_PROVIDER + "/")
        if not model:
            raise ValueError("baseline needs a model name")
        home = "/logs/agent/pi"
        session = home + "/sessions/baseline.jsonl"
        env = {"PI_CODING_AGENT_DIR": home + "/config", "PI_OFFLINE": "1"}
        await self.exec_as_agent(environment, command=f"mkdir -p {home}/sessions {home}/config && "
                                                     f'printf "%s\\n" "$HOME" > {home}/home.txt && '
                                                     f"pi --version > {home}/version.txt")
        models = pi_models(f"http://127.0.0.1:{runtime.PASSTHROUGH_PORT}/v1", model, REASONING_PROVIDER)
        await self._upload_config_text(environment, content=json.dumps(models),
                                       remote_path=home + "/config/models.json", filename="models.json")
        args = pi_args(session, model, REASONING_PROVIDER, instruction)
        await self.exec_as_agent(environment, command=shlex.join(["pi", *args]) +
                                 f" </dev/null >/logs/agent/{self._OUTPUT_FILENAME} 2>&1", env=env)
        await self.exec_as_root(environment, command=shlex.join([
            runtime.RUNTIME, f"{runtime.PROXY_DIR}/src/pi_receipts.py", session, runtime.PASSTHROUGH_LOGDIR]))
