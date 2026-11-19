"""Common IR bookkeeping and sandbox seeding for line-oriented native sessions."""
from collections import Counter, defaultdict
import json
import shlex
from pathlib import Path

from ..trace.ir import _sha256, _ts
from .base import AgentAdapter, SnapshotResumeAgentMixin, bounded_remote_command


NATIVE_REPLAY_REVISION = 6


def mark_parallel_writes(steps, agent):
    for step in steps:
        writers = [e for e in step["events"] if e["type"] == "tool_call" and e["side_effect"] != "read_only"]
        if len(writers) > 1:
            lines = ", ".join(str(e["line"]) for e in writers)
            reason = f"parallel {agent} calls may write (lines {lines}); final workspace order is unknown"
            prior = writers[0].get("replay_refusal")
            writers[0]["replay_refusal"] = f"{prior}; {reason}" if prior else reason


class NativeAdapter(AgentAdapter):
    resume_artifacts = {"prompt": "sentinel_user_message"}

    @staticmethod
    def validate_model(meta, model):
        if model and model not in (meta.get("native_model"), f"{meta.get('native_provider')}/{meta.get('native_model')}"):
            raise ValueError("model override conflicts with the native bundle model")

    def session_relative_path(self, meta):
        relative = Path(meta["session_relpath"])
        if (".." in relative.parts or relative.is_absolute() or relative.suffix != ".jsonl"
                or not relative.is_relative_to(self.session_root)):
            raise ValueError("invalid native session path")
        return relative.relative_to("agent")

    def replay_session_path(self, meta):
        return str(Path("/logs/agent") / self.session_relative_path(meta))

    def find_session(self, trial_dir):
        paths = sorted(Path(trial_dir).glob(self.session_glob))
        if len(paths) != 1:
            raise ValueError(f"expected one {self.name} session, found {len(paths)}")
        return paths[0]

    def cut_line(self, raw_lines, ir, k_index):
        return ir["steps"][k_index]["line_range"][0]

    def native_reference(self, trial, ir, k, prefix):
        path = trial / "agent/model-proxy/native-reference.jsonl"
        entry = ir["steps"][k].get("message_id")
        refs = []
        if entry and path.exists():
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["session_id"] == ir["session_id"] and row["entry_id"] == entry:
                        refs.append(row["content"])
        if len(refs) != 1 or refs[0]["wire"] != self.wire or refs[0]["complete"] is not True:
            return None, "no unique complete captured request for this native entry"
        return dict(content=refs[0], method="captured-entry-id", index=k), None

    def trace(self, path, trial, records, sid, cwd, version, steps, timeout, flags=None, **fields):
        start = records[0]["timestamp"]
        flags = list(flags or [])
        if not (trial / "agent/trajectory.json").is_file() and any("atif_step_id" not in s for s in steps):
            # No Harbor ATIF export: the 1-based position of each step is its id (as for Claude Code).
            flags.append("atif_missing_step_ids_positional")
        for i, step in enumerate(steps):
            elapsed = round(step.get("request_start_s", _ts(step["timestamp"])) - _ts(start), 3)
            step.update(index=i, elapsed_s=elapsed, remaining_s=timeout-elapsed)
            step.setdefault("atif_step_id", i+1)
            step["line_range"].append(steps[i+1]["line_range"][0] if i+1 < len(steps) else len(records))
        return dict(native_replay_revision=NATIVE_REPLAY_REVISION, trial=trial.name, session_id=sid, cwd=cwd, project_slug=cwd.replace("/", "-"),
                    cc_version=version, start_ts=start, flags=flags, sidechains=[], steps=steps,
                    sources=dict(session_jsonl=str(path.relative_to(trial)), session_sha256=_sha256(path)),
                    **fields)


def align_atif_step_ids(trial, steps, call_ids, sources):
    """Bind native steps to Harbor ATIF step ids when ``agent/trajectory.json`` exists.

    A step is bound by the ATIF step that owns its tool call ids, else by (source, timestamp). An unbound or ambiguous
    step gets ``atif_step_id = None`` so ``bundle --step`` refuses it instead of resolving to a positional guess.
    ``call_ids[i]`` lists step i's tool call ids; ``sources[i]`` is its ATIF source. Returns flags."""
    atif = trial / "agent/trajectory.json"
    if not atif.is_file():
        return []
    exported = json.loads(atif.read_text()).get("steps") or []
    by_call = defaultdict(set)
    for s in exported:
        for call in s.get("tool_calls") or []:
            by_call[call["tool_call_id"]].add(s["step_id"])
    by_message = {}
    for s in exported:
        by_message.setdefault((s.get("source"), s.get("timestamp")), []).append(s["step_id"])
    candidates = []
    for step, calls, source in zip(steps, call_ids, sources):
        ids = set().union(*(by_call[c] for c in calls))
        if not ids:
            timestamp = step.get("atif_message_timestamp", step["timestamp"])
            if timestamp:
                ids.update(by_message.get((source, timestamp), []))
        candidates.append(ids)
    owners = Counter(i for ids in candidates for i in ids)
    for step, ids in zip(steps, candidates):
        candidate = next(iter(ids)) if len(ids) == 1 else None
        step["atif_step_id"] = candidate if owners[candidate] == 1 else None
    unbound = sum(step["atif_step_id"] is None for step in steps)
    return [f"atif_unbound_steps:{unbound}"] if unbound else []


class NativeResumeMixin(SnapshotResumeAgentMixin):
    async def install(self, environment):
        bundle, meta = self._bundle()
        version = await self.exec_as_agent(environment, command=f"{self.adapter.name} --version")
        fields = (version.stdout or "").strip().split()
        if (version.return_code or not fields or fields[-1] != meta["agent_version_orig"]
                or meta["agent"] != self.adapter.name):
            raise RuntimeError("snapshot CLI version differs from original")
        self.adapter.sanity_check_bundle(bundle, meta)
        (self._resume_logs() / "bundle.json").write_text(json.dumps(meta))
        await self._start_proxies(environment, bundle, meta)

    async def run(self, instruction, environment, context):
        bundle, meta = self._bundle()
        self.adapter.validate_model(meta, getattr(self, "model_name", None))
        self.adapter.sanity_check_bundle(bundle, meta)
        if self.adapter.name == "pi":
            plan = json.loads((bundle / "plan.json").read_bytes())
            paths = {a["args"]["pi_output_file"] for a in plan["actions"] if a["args"].get("pi_output_file")}
            manifest = getattr(getattr(self, "options", None), "pi_output_manifest", None) or {}
            if set(manifest) != paths:
                raise RuntimeError("missing Pi cut output manifest for this plan")
            for path, info in manifest.items():
                quoted = shlex.quote(path)
                if info is None:
                    command = f"test ! -e {quoted} && test ! -L {quoted}"
                else:
                    ownership = f"{int(info['uid'])}:{int(info['gid'])}:{int(info['mode']):o}"
                    expected = shlex.quote(info["sha256"] + "  " + path)
                    command = (f'test ! -L {quoted} && test -f {quoted} && '
                               f'test "$(stat -c %u:%g:%a {quoted})" = {ownership} && '
                               f'test "$(sha256sum {quoted})" = {expected}')
                await self.exec_as_agent(environment, command=command)
        env, remote, args, config_path, config = self.adapter.resume_setup(meta, self._resume_base_url(meta["run_id"]))
        dirs = {str(Path(remote).parent), str(Path(config_path).parent)}
        await self.exec_as_agent(environment, command="mkdir -p " + shlex.join(sorted(dirs)), env=env)
        local = self._resume_logs() / Path(config_path).name
        local.write_text(config)
        seeded = {remote: bundle / "session_trunc.jsonl", config_path: local}
        if meta["aux_counts"].get("model_catalog"):
            src = bundle / "aux/model-catalog.json"
            seeded[str(Path(config_path).parent / src.name)] = src
        for dst, src in seeded.items():
            await self._upload_session_file(environment, src, dst)
        await self._verify_seeded(environment, seeded, env)
        output = self.environment_logs_dir / self._OUTPUT_FILENAME
        command = f"cd {shlex.quote(meta['cwd'])} && {shlex.join(args)} </dev/null >{shlex.quote(str(output))} 2>&1"
        if self.adapter.name == "pi":
            from .pi import check_output_command
            command += " && " + check_output_command(output, resumed=True)
        await self.exec_as_agent(environment, command=bounded_remote_command(command, meta["agent_budget_s"]), env=env)


class NativeOptions:
    pi_output_manifest: dict | None = None
    resume_bundle: str | None = None
    upstream_fake: bool = False
    upstream_url: str | None = None
    parity_gate: str = "auto"
    max_model_requests: int | None = None
    freeze_first_request: str | None = None
    expected_first_request_sha256: str | None = None
    passthrough_script: str | None = None
    passthrough_script_sha256: str | None = None
    allow_incomplete_evidence: bool = False
    parity_verdict: str | None = None
    fake_parity_verdict: str | None = None
    passthrough_args: list[str] | None = None
