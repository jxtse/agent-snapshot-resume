"""Codex CLI 0.154.0 rollouts (Responses). One model call can contain parallel tools."""
import hashlib
import json
import math
import posixpath
import re
import shlex
from pathlib import Path

from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.models.trajectories import FinalMetrics

from ..proxy.native_wire import digest
from ..proxy.runtime import fetch
from ..trace.ir import _ts, bash_is_read_only
from .native import align_atif_step_ids, NativeAdapter, NativeOptions, NativeResumeMixin


CATALOG_SHA256 = "b7d04bd6ed29640ac4a80cd6ffbda125d94dd5f8ccb95d29721099d4118ec873"
DEFAULT_CATALOG = Path(".cache/deepseek-models.json")


def read_catalog(path, model):
    """Verify the exact bytes used for both seeding and selecting the model policy."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ValueError(f"missing pinned Codex model catalog: {path}") from None
    if hashlib.sha256(data).hexdigest() != CATALOG_SHA256:
        raise ValueError(f"Codex model catalog SHA256 mismatch: {path}")
    policies = [entry["truncation_policy"] for entry in json.loads(data)["models"] if entry["slug"] == model]
    if len(policies) != 1 or policies[0]["mode"] != "tokens" or policies[0]["limit"] <= 0:
        raise ValueError(f"no supported token policy for model {model!r} in pinned catalog")
    return data, policies[0]["limit"]


def render_text(text, tokens):
    data = text.encode("utf-8")
    budget = 4 * tokens
    if len(data) <= budget:
        return text
    half = budget // 2
    head = data[:half].decode("utf-8", errors="ignore")
    tail = data[len(data) - half:].decode("utf-8", errors="ignore")
    omitted = (len(data) - budget + 3) // 4
    return f"{head}…{omitted} tokens truncated…{tail}"


def render_output(output, catalog_limit):
    # Context history grants tool outputs 120% of the catalog token limit in 0.154.0.
    remaining = math.ceil(catalog_limit * 1.2)
    if isinstance(output, str):
        return render_text(output, remaining)
    blocks = []
    omitted = 0
    for block in output:
        if block["type"] != "input_text":
            blocks.append(block)
            continue
        text = block["text"]
        if not text:
            continue
        if remaining:
            blocks.append({**block, "text": render_text(text, remaining)})
            remaining = max(0, remaining - (len(text.encode("utf-8")) + 3) // 4)
        else:
            omitted += 1
    if omitted:
        blocks.append({"type": "input_text", "text": f"[omitted {omitted} text items ...]"})
    return blocks


def project_item(item, catalog_limit):
    projected = {k: v for k, v in item.items() if k != "internal_chat_message_metadata_passthrough"}
    if projected.get("type") in {"function_call_output", "custom_tool_call_output"}:
        projected["output"] = render_output(projected["output"], catalog_limit)
    return projected


def codex_history(records, catalog_limit):
    """Projected model history. Keep wire IDs and nulls. Exclude storage-only turn metadata."""
    history = []
    for row in records:
        payload = row["payload"]
        if row["type"] == "response_item":
            history.append(payload)
        elif row["type"] == "compacted":
            replacement = payload.get("replacement_history")
            if not isinstance(replacement, list):
                raise ValueError("compaction has no explicit replacement history")
            history = list(replacement)
    return [project_item(item, catalog_limit) for item in history]


def recorded_sandbox_mode(context):
    policy = context.get("sandbox_policy")
    if policy is None:
        return None  # Older rollout schemas carried no restorable policy.
    if (policy == {"type": "danger-full-access"} and context.get("approval_policy") == "never"
            and context.get("permission_profile") == {"type": "disabled"}):
        return "danger-full-access"
    if policy == {"type": "read-only"} and context.get("approval_policy") in (None, "never"):
        return "read-only"
    return "unsupported"


# A literal range-print script with one plain file operand cannot edit the file.
# Keep this native-only so historical Claude Code plans retain their exact actions.
_SED_PRINT = re.compile(r"""(?:/usr/bin/|/bin/)?sed -n '(?:[0-9]+,)?[0-9]+p' [a-zA-Z0-9_./][a-zA-Z0-9_./-]*""")


def tool_action(payload, cwd):
    name = payload["name"]
    if payload["type"] == "custom_tool_call":
        if name == "apply_patch":
            # The pinned executable provides the patch implementation, including moves and contextual hunks.
            return "Bash", {"command": shlex.join(["codex", "--codex-run-as-apply-patch", payload["input"]])}, "exec", cwd
        return name, {"input": payload["input"]}, "other", cwd
    args = json.loads(payload["arguments"])
    if name in ("exec_command", "shell", "shell_command"):
        cmd = args.get("cmd", args.get("command"))
        if isinstance(cmd, list):
            cmd = shlex.join(cmd)
        shell = args.get("shell")
        if isinstance(cmd, str) and not args.get("tty") and (shell is None or isinstance(shell, str)):
            standard_shell = shell in (None, "bash", "/bin/bash", "sh", "/bin/sh")
            read_only = standard_shell and (bash_is_read_only(cmd) or _SED_PRINT.fullmatch(cmd))
            effect = "read_only" if read_only else "exec"
            if shell or args.get("login") is False:
                cmd = shlex.join([shell or "bash", "-lc" if args.get("login", True) else "-c", cmd])
            return ("Bash", {"command": cmd}, effect,
                    posixpath.join(cwd, args.get("workdir") or cwd))
    return name, args, "other", cwd


def decode_output(output, *, custom=None):
    if not isinstance(output, str):
        return json.dumps(output, ensure_ascii=False), None, None, None
    if custom is not False:
        header = re.match(r"\AExit code: (-?\d+)\nWall time: (\d+(?:\.\d+)?)(?: seconds)?\nOutput:\n", output)
        if header:
            return output[header.end():], int(header[1]), None, float(header[2])
    if output.startswith("Chunk ID:"):
        marker = re.search(r"(?m)^(?:Final output|Output):(?:\n|$)", output)
        if marker:
            header, text = output[:marker.start()], output[marker.end():]
            status = re.search(r"(?m)^Process exited with code (-?\d+)$", header)
            session = re.search(r"(?m)^Process running with session ID (\d+)$", header)
            wall = re.search(r"(?m)^Wall time: ([.\d]+) seconds", header)
            return (text, int(status[1]) if status else None, session[1] if session else None,
                    float(wall[1]) if wall else None)
    return output, None, None, None


class CodexAdapter(NativeAdapter):
    name, version, wire = "codex", "0.154.0", "responses"
    resume_artifacts = {**NativeAdapter.resume_artifacts, "resume_context": "codex-0.154.0-skills-permissions-environment"}

    def __init__(self, catalog_path=DEFAULT_CATALOG):
        self.catalog_path = Path(catalog_path)

    def trial_catalog(self, agent_dir, model):
        exported = agent_dir / "model-catalog.json"
        return read_catalog(exported if exported.exists() or exported.is_symlink() else self.catalog_path, model)

    def bundle_artifacts(self, cwd, original=None):
        artifacts = {**self.resume_artifacts, "cwd": cwd, "codex_home": "/logs/agent"}
        if original and original.get("content"):
            artifacts["prefix_items"] = len(original["content"]["items"])
        return artifacts

    session_root = "agent/sessions"
    session_glob = session_root + "/*/*/*/rollout-*.jsonl"
    binary_sha256 = {"0.154.0": "e27c83a49e6031685ee7f956c12aad5f16484d3a80181dd3fea930fb96b3832b"}
    harbor_import_path = "agent_snapshot_resume.agents.codex:CodexSnapshotResume"

    def load_trace(self, trial_dir, agent_timeout_s, task=None, reward=None):
        trial = Path(trial_dir)
        path = self.find_session(trial)
        with path.open("rb") as stream:
            rows = [json.loads(line) for line in stream]
        header = rows[0]["payload"]
        if rows[0]["type"] != "session_meta" or header["cli_version"] != self.version:
            raise ValueError("unsupported Codex rollout version")
        steps, calls = [], {}
        current = last_counted = None
        model, cwd, context = None, header["cwd"], {}
        processes = {}
        for i, row in enumerate(rows):
            typ, p, ts = row["type"], row["payload"], row["timestamp"]
            subtype = p.get("type")
            if typ == "turn_context":
                current = last_counted = None
                context = p
                model, cwd = p.get("model"), p["cwd"]
            if typ == "compacted" and steps:
                steps[-1]["events"].append(dict(type="compaction_boundary", line=i))
                current = last_counted = None
            if typ == "token_usage_record":
                owner = current or last_counted
                if owner is not None:
                    owner.update(usage=p["usage"], message_id=p.get("response_id"))
                current = last_counted = None  # subsequent tool results still bind via call ID
            if typ == "event_msg" and subtype == "token_count":
                if current is not None:
                    last_counted = current
                    last = (p.get("info") or {}).get("last_token_usage")
                    if last:
                        current["usage"] = last
                current = None
            if typ != "response_item":
                continue
            if subtype == "message" and p["role"] in ("developer", "system", "user"):
                current = last_counted = None
                steps.append(dict(source="user", atif_source={"developer": "system"}.get(p["role"], p["role"]), line_range=[i], timestamp=ts,
                                  events=[dict(type="user_text", text=p["content"])]))
                continue
            if subtype in ("function_call_output", "custom_tool_call_output"):
                event, step = calls[p["call_id"]]
                text, rc, session_id, duration = decode_output(p["output"], custom=subtype == "custom_tool_call_output")
                if subtype == "custom_tool_call_output" and event["tool"] == "Bash" and rc is not None:
                    event["args"]["codex_exit_code"] = rc
                if rc not in (None, 0) and event["tool"] == "Bash":
                    text = f"Exit code {rc}\n" + text
                event.update(is_error=rc != 0 if rc is not None else None, duration_s=duration)
                if session_id and event["tool"] == "Bash":
                    # A PTY session cannot be revived by executing a fresh foreground command.
                    event.update(tool="unified_exec", side_effect="exec_background", background_task_id=session_id)
                    processes[session_id] = event["call_id"]
                if event["tool"] == "write_stdin" and rc is not None:
                    sid = str(event["args"]["session_id"])
                    if sid in processes:
                        step["events"].append(dict(type="attachment", bg_task_id=sid, bg_call_id=processes.pop(sid),
                                                   bg_status="completed" if rc == 0 else "failed", bg_exit_code=rc,
                                                   timestamp=ts, line=i))
                step["events"].append(dict(type="observation", call_id=p["call_id"], text=text, line=i,
                                           timestamp=ts, is_error=event["is_error"]))
                continue
            if current is None:
                last_counted = None
                current = dict(source="agent", timestamp=ts, atif_message_timestamp=None, line_range=[i], events=[],
                               native_model=model, native_cwd=cwd, native_effort=context.get("effort"),
                               native_sandbox_mode=recorded_sandbox_mode(context))
                steps.append(current)
            if subtype in ("function_call", "custom_tool_call"):
                tool, args, effect, workdir = tool_action(p, cwd)
                event = dict(type="tool_call", tool=tool, args=args, side_effect=effect, cwd=workdir,
                             call_id=p["call_id"], line=i, timestamp=ts)
                permissions = (json.loads(p["arguments"]).get("sandbox_permissions")
                               if subtype == "function_call" else None)
                policy = recorded_sandbox_mode(context)
                if effect != "read_only" and (policy != "danger-full-access" or permissions not in (None, "use_default")):
                    event["replay_refusal"] = f"Codex sandbox policy {policy or 'unrecorded'} cannot be reproduced"
                calls[p["call_id"]] = event, current
            elif subtype in ("message", "reasoning"):
                if subtype == "message":
                    current["atif_message_timestamp"] = current["atif_message_timestamp"] or ts
                event = dict(type="assistant_text" if subtype == "message" else "assistant_reasoning",
                             text="\n".join(b.get("text", "") for b in p.get("content") or []))
            else:
                raise ValueError(f"unsupported Codex response item {subtype!r}")
            current["events"].append(event)
        for st in steps:
            writers = [e for e in st["events"] if e["type"] == "tool_call" and e["side_effect"] != "read_only"]
            if len(writers) > 1:
                lines = ", ".join(str(e["line"]) for e in writers)
                reason = f"parallel Codex calls may write (lines {lines}); final workspace order is unknown"
                writers[0]["replay_refusal"] = "; ".join(filter(None, [writers[0].get("replay_refusal"), reason]))
        flags = align_atif_step_ids(
            trial, steps, [[e["call_id"] for e in st["events"] if e["type"] == "tool_call"] for st in steps],
            [st.get("atif_source", st["source"]) for st in steps])
        records = self.request_records(trial)
        for st in steps:
            if st["source"] != "agent":
                continue
            hits, _ = self.match_records(records, st, header["id"])
            if len(hits) == 1 and hits[0][1].get("ts"):
                value = hits[0][1]["ts"]
                try:
                    ts = float(value) if type(value) in (float, int) else _ts(value)
                except (TypeError, ValueError, AttributeError):
                    continue
                if ts is not None and math.isfinite(ts) and ts <= _ts(st["timestamp"]):
                    st["request_start_s"] = max(ts, _ts(rows[0]["timestamp"]))
        return self.trace(path, trial, rows, header["id"], header["cwd"], self.version, steps, agent_timeout_s,
                          flags=flags, task=task, reward=reward, model=model, provider=header["model_provider"])

    def cut_line(self, raw_lines, ir, k_index):
        cut = super().cut_line(raw_lines, ir, k_index)
        while cut > 0:
            row = json.loads(raw_lines[cut - 1])
            p = row["payload"]
            if (row["type"] == "event_msg" and p.get("type") == "item_completed" and
                    p.get("item", {}).get("type") in {"Reasoning", "AgentMessage"}):
                cut -= 1
            else:
                break
        return cut

    @staticmethod
    def request_records(trial):
        records = []
        for path in sorted((trial/"agent/model-proxy").glob("requests-*.jsonl")):
            with path.open("rb") as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
        return records

    @staticmethod
    def match_records(records, step, session_id):
        usage = step.get("usage") or {}
        entry = step.get("message_id")
        model = step.get("native_model")
        identities = {"session": session_id, "session_id": session_id, "thread_id": session_id,
                      "response_id": entry, "entry_id": entry, "model": model}
        records = [(i, row) for i, row in enumerate(records)
                   if all(not row.get(key) or expected is None or row[key] == expected
                          for key, expected in identities.items())]
        captured = [(i, r) for i, r in records if entry and r.get("entry_id") == entry
                    and r.get("session_id") == session_id and r.get("status") == 200 and r.get("native_reference")]
        if captured:
            return captured, True
        keys = ("input_tokens", "output_tokens")
        hits = [(i, r) for i, r in records if r.get("status") == 200 and
                all(usage.get(key) is not None and usage[key] == (r.get("usage") or {}).get(key) for key in keys)]
        return hits, False

    def native_reference(self, trial, ir, k, prefix):
        if (trial / "agent/model-proxy/native-reference.jsonl").exists():
            return super().native_reference(trial, ir, k, prefix)
        model = ir["steps"][k]["native_model"]
        hits, captured = self.match_records(self.request_records(trial), ir["steps"][k], ir["session_id"])
        if captured:
            if len(hits) != 1:
                return None, "ambiguous captured Codex request"
            i, record = hits[0]
            ref = record["native_reference"]
            if ref.get("wire") != self.wire or ref.get("complete") is not True:
                return None, "invalid captured Codex reference"
            return dict(index=i, method="captured-entry-id", content=ref), None
        if len(hits) != 1:
            return None, "no unique Codex usage match"
        index, record = hits[0]
        rows = [json.loads(l) for l in prefix]
        _, limit = self.trial_catalog(trial / "agent", ir["steps"][k]["native_model"])
        history = codex_history(rows, limit)
        instructions = rows[0]["payload"]["base_instructions"]["text"]
        content = dict(wire=self.wire, complete=False, items=[digest(item) for item in history],
                       fields={"instructions": digest(instructions), "model": digest(record.get("model") or model)},
                       projection="codex_rollout_tokens", missing_evidence=["original_request_envelope"], logged_item_count=record.get("n_items"))
        return dict(index=index, method="unique-usage", content=content,
                    metadata={k: record.get(k) for k in ("req_bytes", "n_items", "n_tools", "effort", "n_msgs",
                                                      "n_reasoning_replayed", "dropped_tools", "dropped_fields") if k in record}), None

    def collect_aux(self, session_path, ir, k_index, prefix_text, out_dir):
        model = ir["steps"][k_index]["native_model"]
        content, _ = self.trial_catalog(session_path.parents[4], model)
        (out_dir / "model-catalog.json").write_bytes(content)
        return {"model_catalog": 1}

    def sanity_check_bundle(self, bundle_dir, meta):
        if meta.get("aux_counts", {}).get("model_catalog") != 1:
            raise ValueError("Codex resume requires a bundled, pinned model catalog")
        read_catalog(bundle_dir / "aux/model-catalog.json", meta["native_model"])
        session = bundle_dir / "session_trunc.jsonl"
        if session.is_file():
            policy = cwd = initial_cwd = None
            with session.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["type"] == "session_meta":
                        initial_cwd = cwd = row["payload"]["cwd"]
                    elif row["type"] == "turn_context":
                        cwd = row["payload"]["cwd"]
                        policy = recorded_sandbox_mode(row["payload"])
            if cwd != meta.get("cwd") or initial_cwd != meta.get("initial_cwd", initial_cwd):
                raise ValueError("stale Codex working directory; rebuild the bundle")
            if policy == "unsupported" or policy != meta.get("native_sandbox_mode"):
                raise ValueError("unsupported or stale Codex sandbox policy; rebuild the bundle")

    def install_commands(self, version=None):
        v = version or self.version
        if v not in self.binary_sha256:
            raise ValueError("unsupported Codex version")
        checksum = self.binary_sha256[v]
        url = f"https://registry.npmjs.org/@openai/codex/-/codex-{v}-linux-x64.tgz"
        directory = "/opt/asr-codex/package/vendor/x86_64-unknown-linux-musl"
        cmd = (fetch(url, "/tmp/asr-codex.tgz", checksum) +
               " && mkdir -p /opt/asr-codex && tar -xzf /tmp/asr-codex.tgz -C /opt/asr-codex"
               f" && (for exe in {directory}/bin/codex {directory}/codex/codex; do "
               'if test -x "$exe"; then ln -sf "$exe" /usr/local/bin/codex && exit; fi; done; exit 1)'
               f" && test \"$(codex --version)\" = 'codex-cli {v}'")
        return [("codex", cmd, 300)]

    def resume_setup(self, meta, base, home="/logs/agent"):
        relative = self.session_relative_path(meta)
        provider, model = meta["native_provider"], meta["native_model"]
        if meta.get("aux_counts", {}).get("model_catalog") != 1:
            raise ValueError("Codex resume requires a bundled, pinned model catalog")
        catalog = f'model_catalog_json = {json.dumps(str(Path(home) / "model-catalog.json"))}\n'
        mode = meta.get("native_sandbox_mode")
        if mode not in (None, "read-only", "danger-full-access"):
            raise ValueError("unsupported Codex sandbox policy")
        policy = f'sandbox_mode = {json.dumps(mode)}\napproval_policy = "never"\n' if mode else ""
        config = (policy + catalog + f'model = {json.dumps(model)}\nmodel_provider = {json.dumps(provider)}\n'
                  f'model_reasoning_effort = {json.dumps(meta.get("native_effort") or "max")}\n'
                  'web_search = "disabled"\nshow_raw_agent_reasoning = true\n'
                  '[features]\nplugins = false\nremote_plugin = false\n'
                  '[analytics]\nenabled = false\n[feedback]\nenabled = false\n'
                  f'[model_providers.{json.dumps(provider)}]\nname = {json.dumps(provider)}\n'
                  f'base_url = {json.dumps(base + "/v1")}\nwire_api = "responses"\n'
                  'env_key = "ASR_DUMMY_KEY"\nrequest_max_retries = 0\nstream_max_retries = 0\n')
        return ({"CODEX_HOME": home, "ASR_DUMMY_KEY": "dummy"}, str(Path(home) / relative),
                ["codex", "exec", "resume", "--json", "--skip-git-repo-check", meta["session_id"], meta["sentinel"]],
                home + "/config.toml", config)


class CodexSnapshotResumeOptions(NativeOptions, CodexOptions):
    pass


class CodexSnapshotResume(NativeResumeMixin, Codex):
    options_model = CodexSnapshotResumeOptions
    adapter = CodexAdapter()

    def _convert_events_to_trajectory(self, session_dir):
        trajectory = super()._convert_events_to_trajectory(session_dir)
        if trajectory is None:
            return None
        # Harbor persists all steps and bills FinalMetrics. Replace only the latter.
        trajectory.final_metrics = None
        bundle, meta = self._bundle()
        session = session_dir / Path(meta["session_relpath"]).name
        if not session.is_file() or session != max(session_dir.glob("*.jsonl")):
            return trajectory
        with session.open("rb") as stream, (bundle / "session_trunc.jsonl").open("rb") as seed:
            for block in iter(lambda: seed.read(1024 * 1024), b""):
                if stream.read(len(block)) != block:
                    return trajectory
            calls = []
            seen = {}
            pending_output = False
            model = meta.get("native_model")
            for line in stream:
                row = json.loads(line)
                payload = row["payload"]
                kind = row["type"]
                if kind == "turn_context":
                    model = payload.get("model", model)
                elif kind == "response_item":
                    if (payload.get("role") == "assistant"
                            or payload.get("type") in {"function_call", "custom_tool_call", "reasoning"}):
                        pending_output = True
                elif kind == "token_usage_record":
                    usage = payload["usage"]
                    rid = payload.get("response_id")
                    if rid and rid in seen:
                        if seen[rid] != usage:
                            return trajectory
                    else:
                        calls.append((model, usage))
                        if rid:
                            seen[rid] = usage
                    pending_output = False
                elif kind == "event_msg" and payload.get("type") == "token_count" and pending_output:
                    # Some exports carry only Harbor's last-call counter. Bind it to
                    # new model output, since token_count can also repeat the previous usage.
                    usage = (payload.get("info") or {}).get("last_token_usage")
                    if not usage:
                        return trajectory
                    calls.append((model, usage))
                    pending_output = False
        if not calls or pending_output:
            return trajectory
        costs = [self._compute_cost_from_pricing(
            prompt_tokens=u["input_tokens"], completion_tokens=u["output_tokens"],
            cached_tokens=u.get("cached_input_tokens"), cache_write_tokens=u.get("cache_write_input_tokens"),
            model_name=m) for m, u in calls]
        trajectory.final_metrics = FinalMetrics(
            total_prompt_tokens=sum(u["input_tokens"] for _, u in calls),
            total_completion_tokens=sum(u["output_tokens"] for _, u in calls),
            total_cached_tokens=sum(u.get("cached_input_tokens", 0) for _, u in calls),
            total_cost_usd=sum(costs) if all(cost is not None for cost in costs) else None,
            total_steps=len(trajectory.steps),
            extra={"resumed_calls": len(calls),
                   "reasoning_output_tokens": sum(u.get("reasoning_output_tokens", 0) for _, u in calls),
                   "total_cache_write_input_tokens": sum(u.get("cache_write_input_tokens", 0) for _, u in calls)})
        return trajectory


def resolve_polled_commands(plan, steps):
    """Replay a completed foreground command once when its only intervening calls were empty polls."""
    actions = plan["actions"]
    events = sorted((e for step in steps for e in step["events"] if e["type"] == "tool_call"),
                    key=lambda e: e["line"])
    by_call = {a["call_id"]: a for a in actions}
    consumed = set()
    for i, event in enumerate(events):
        if event["tool"] != "unified_exec":
            continue
        start = by_call[event["call_id"]]
        start["side_effect"] = "other"
        if not start.get("bg_finished_before_k"):
            continue
        polls = []
        for next_event in events[i+1:]:
            args = next_event["args"]
            if (next_event["tool"] != "write_stdin" or args.get("chars", "") != ""
                    or str(args.get("session_id")) != start["background_task_id"]
                    or next_event["line"] > start["bg_finished_line"]):
                break
            polls.append(by_call[next_event["call_id"]])
        if not polls or polls[-1]["obs_line"] != start["bg_finished_line"]:
            continue
        start.update(tool="Bash", side_effect="exec", duration_s=start.get("bg_duration_s"),
                     is_error=polls[-1]["is_error"], orig_text="".join(a["orig_text"] for a in [start, *polls]))
        if start["is_error"]:
            start["orig_text"] = f"Exit code {start['bg_exit_code']}\n" + start["orig_text"]
        consumed.update(p["call_id"] for p in polls)
    plan["actions"] = [a for a in actions if a["call_id"] not in consumed]
