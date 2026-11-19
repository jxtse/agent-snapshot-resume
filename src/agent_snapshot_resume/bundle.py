"""Build a resume bundle from a Harbor trial dir and a step k (host side, read-only on the trial).

Bundle layout (<out>/<run_id>/):
  bundle.json            metadata: trial, session id, project slug, cwd, cut_line, agent budget, sentinel, ...
  session_trunc.jsonl    the native session's lines[:cut], byte-exact (ends right before step k)
  plan.json              replay plan: side-effecting actions before step k with their original observations
  aux/...                adapter-specific auxiliary files (Claude Code: aux/tool-results/*, aux/tasks/*.json)
  proxy_spec.json        resume-proxy spec (no secrets): mode, sentinel, hint, pin_date, resume artifacts

The remaining agent budget is  agent_timeout - elapsed(k),  where agent_timeout is the task's agent timeout
(--agent-timeout, else [agent].timeout_sec of --task-dir/task.toml, else of <trial>/task.toml).
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import re
import shlex
import shutil
import tomllib
from pathlib import Path
from typing import Any

from . import chain, parity
from .trace import ir as trace_ir

SENTINEL_PREFIX = "ZZZ_RESUME_SENTINEL_"


def resolve_agent_timeout(trial_dir: Path, agent_timeout: float | None = None,
                          task_dir: Path | None = None) -> float:
    if agent_timeout:
        return float(agent_timeout)
    for cand in ([task_dir / "task.toml"] if task_dir else []) + [trial_dir / "task.toml",
                                                                  trial_dir / "config" / "task.toml"]:
        if cand.is_file():
            t = ((tomllib.loads(cand.read_text()).get("agent") or {}).get("timeout_sec"))
            if t:
                return float(t)
    raise SystemExit("agent timeout unknown: pass --agent-timeout SECONDS or --task-dir <task with task.toml>")


def _trial_result(trial_dir: Path) -> dict[str, Any]:
    p = trial_dir / "result.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def default_run_id(trial: str, atif_step: int, mode: str) -> str:
    rid = f"{trial.split('__')[0][:28]}-{trial.split('__')[-1][:8]}-s{atif_step}-{mode}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", rid)


def sentinel_for(run_id: str) -> str:
    return SENTINEL_PREFIX + hashlib.sha1(run_id.encode()).hexdigest()[:10]


def snapshot_cut_identity(bundle_dir: Path) -> dict[str, Any]:
    """What a step-k environment snapshot depends on. Bundles with equal identity (e.g. the mode=same and mode=hint
    bundles of one trial at one k) can share one snapshot: the replay plan, cut, fingerprint roots and agent version
    are identical; only the session seed / proxy spec (uploaded at launch, not part of the snapshot) differ."""
    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / "bundle.json").read_text())
    return {"trial": meta["trial"], "session_sha256": meta["session_sha256"], "cut_line": meta["cut_line"],
            "resume_atif_step": meta["resume_atif_step"], "agent": meta.get("agent"),
            "base_snapshot": (meta.get("base_snapshot") or {}).get("snapshot_id"),
            "agent_version_orig": meta.get("agent_version_orig"), "fingerprint_roots": meta.get("fingerprint_roots"),
            "fingerprint_optional_roots": meta.get("fingerprint_optional_roots"),
            "plan_sha256": hashlib.sha256((bundle_dir / "plan.json").read_bytes()).hexdigest(),
            "replay_reviewed_exceptions": meta.get("replay_reviewed_exceptions") or {}}


def fingerprint_command_optional_roots(command: str | None) -> list[str] | None:
    """Optional fingerprint roots baked into a recorded fingerprint command (`optional=json.loads('<json>')`).
    [] when the command predates optional roots (it could not have tolerated a missing root, so nothing was
    optional), None when the command cannot be read at all."""
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    code = parts[-1] if parts else ""
    marker = "optional=json.loads("
    i = code.find(marker)
    if i < 0:
        return []
    j = code.find(")", i + len(marker))
    if j < 0:
        return None
    try:
        return list(json.loads(ast.literal_eval(code[i + len(marker):j])))
    except (ValueError, SyntaxError, TypeError):
        return None


def cut_identity_diff(mine: dict[str, Any], theirs: dict[str, Any],
                      recorded_fingerprint_command: str | None = None) -> list[str]:
    """Fields in which a prepared snapshot's recorded cut identity differs from `mine`.

    `fingerprint_optional_roots` is compared with the absent-optional-field compatibility rule: a state written
    before that field existed (key missing, or null) is identical to a bundle that names no optional roots, and a
    state that records its fingerprint command must show that the command used no optional roots. A snapshot that
    tolerated a missing root is a *weaker* fingerprint than one that requires every root, so this widening never
    hides a file difference; a bundle that *adds* optional roots is still refused."""
    diff = []
    for k in mine:
        if theirs.get(k) == mine.get(k):
            continue
        if k == "fingerprint_optional_roots" and not (mine.get(k) or []) and not (theirs.get(k) or []):
            if recorded_fingerprint_command is None or fingerprint_command_optional_roots(
                    recorded_fingerprint_command) == []:
                continue
        diff.append(k)
    return sorted(diff)


def bundle_identity(bundle_dir: Path) -> dict[str, Any]:
    """Everything a resumed run depends on: the snapshot cut identity (replay plan, reviews, fingerprint roots, cut,
    agent version) plus the session seed (truncated session + auxiliary files) and the proxy spec. Two runs with
    equal identity start from the same environment and the same agent state."""
    bundle_dir = Path(bundle_dir)
    seed = hashlib.sha256()
    for f in [bundle_dir / "session_trunc.jsonl"] + sorted(p for p in (bundle_dir / "aux").rglob("*") if p.is_file()):
        seed.update(str(f.relative_to(bundle_dir)).encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    meta = json.loads((bundle_dir / "bundle.json").read_text())
    return {"cut": snapshot_cut_identity(bundle_dir), "seed_sha256": seed.hexdigest(),
            "proxy_spec_sha256": hashlib.sha256((bundle_dir / "proxy_spec.json").read_bytes()).hexdigest(),
            "orig_request_at_k": meta.get("orig_request_at_k"), "run_id": meta.get("run_id")}


# Keys the content reference *derived* after some fake runs were recorded. They describe the same message blocks
# `message_block_sha256` already covers (block kinds of those blocks, digests of the isMeta records), so a fake run
# that predates them judged exactly the same original request. Any other key difference stays a refusal.
ADDITIVE_CONTENT_KEYS = ("block_kinds", "meta_block_sha256")


def orig_request_diff(mine: dict[str, Any] | None, theirs: dict[str, Any] | None) -> list[str]:
    """Fields in which a recorded original-request reference differs from this bundle's (empty == identical)."""
    if not mine or not theirs:
        return ["orig_request_at_k"]
    diff = [k for k in mine if k != "content" and theirs.get(k) != mine.get(k)]
    a, b = mine.get("content") or {}, theirs.get("content") or {}
    for k in set(a) | set(b):
        if k not in a or k not in b:
            if k in ADDITIVE_CONTENT_KEYS:
                continue  # derived, absent on one side: the shared message-block digests decide the content
            if k in a or k in b:
                diff.append("content." + k)
            continue
        if a[k] != b[k]:
            diff.append("content." + k)
    return sorted(diff)


def bundle_identity_diff(mine: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    """Top-level identity fields that differ, with the compatibility rules of `cut_identity_diff` and
    `orig_request_diff` (an older record that lacks only derived content keys is identical)."""
    diff = []
    for k in mine:
        if theirs.get(k) == mine.get(k):
            continue
        if k == "cut" and not cut_identity_diff(mine.get("cut") or {}, theirs.get("cut") or {}):
            continue
        if k == "orig_request_at_k" and not orig_request_diff(mine.get(k), theirs.get(k)):
            continue
        diff.append(k)
    return sorted(diff)


def default_fingerprint_roots(plan: dict[str, Any]) -> list[str] | None:
    """Task directories to fingerprint when none are given. None (= DEFAULT_ROOTS, /app) if the run worked under
    /app or its cwd is unknown. For a run in a home directory (e.g. cwd /root): every first-level, non-hidden
    directory of that home that an action used as cwd or wrote a file into (e.g. /root/data, /root/results,
    /root/work); the whole home is never fingerprinted (caches, agent config)."""
    cwd = plan.get("initial_cwd") or "/app"
    if cwd == "/app" or cwd.startswith("/app/"):
        return None
    parts = [p for p in cwd.split("/") if p]
    home = "/" + "/".join(parts[:2] if parts and parts[0] == "home" else parts[:1])
    paths = [a.get("cwd") for a in plan.get("actions", [])]
    paths += [(a.get("args") or {}).get("file_path") for a in plan.get("actions", [])]
    roots = set()
    for x in paths:
        if not x or not x.startswith(home.rstrip("/") + "/"):
            continue
        first = x[len(home.rstrip("/")) + 1:].split("/")[0]
        if first and not first.startswith("."):
            roots.add(home.rstrip("/") + "/" + first)
    if roots:
        return sorted(roots)
    # A task that works outside /app (e.g. a /task workdir) with no file writes in the plan: bind its own work
    # directory -- fingerprinting /app would fail on an image that has no /app at all. A cwd inside a user's home
    # is never bound as a whole (it carries caches and agent config); such a run keeps the default root and fails
    # its fingerprint step instead of silently hashing the home.
    home_dirs = ("/root", "/home", "/Users")
    return [cwd] if cwd and not any(cwd == h or cwd.startswith(h + "/") for h in home_dirs) else None


def captured_shell_env(trial_dir):
    """Extract a stable literal PATH from CC snapshots, never source arbitrary archived shell code."""
    values = set()
    for path in (Path(trial_dir) / "agent" / "sessions" / "shell-snapshots").glob("*.sh"):
        for line in path.read_text(errors="replace").splitlines():
            match = re.fullmatch(r"(?:export|declare -x) PATH=(.+)", line.strip())
            if match and not re.search(r"[$`\n]", match[1]):
                try:
                    parts = shlex.split(match[1])
                except ValueError:
                    continue
                if len(parts) == 1:
                    values.add(parts[0])
    return {"PATH": values.pop()} if len(values) == 1 else {}


def plan_with_observations(ir: dict[str, Any], k: int) -> dict[str, Any]:
    """Replay plan for steps < k with each action's original observation text attached (`orig_text`)."""
    plan = trace_ir.replay_plan(ir, k)
    obs: dict[str, Any] = {}
    obs_line: dict[str, Any] = {}
    interpretations: dict[str, Any] = {}
    for s in ir["steps"][:k]:
        for e in s["events"]:
            if e["type"] == "observation":
                obs[e["call_id"]] = e["text"]
                obs_line[e["call_id"]] = e.get("line")
                interpretations[e["call_id"]] = e.get("return_code_interpretation")
    for sc in ir["sidechains"]:
        for c in sc["tool_calls"]:
            obs.setdefault(c["call_id"], c.get("observation_text"))
    for a in plan["actions"]:
        a["orig_text"] = obs.get(a["call_id"])
        a["obs_line"] = obs_line.get(a["call_id"])
        if interpretations.get(a["call_id"]):
            a["return_code_interpretation"] = interpretations[a["call_id"]]
        if a.get("side_effect") == "exec_background":
            for step in ir["steps"][:k]:
                for event in step["events"]:
                    args = event.get("args") or {}
                    if (event.get("type") == "tool_call" and event.get("tool") == "Read"
                            and args.get("file_path") == a.get("background_output_file")
                            and not args.get("offset") and not args.get("limit")
                            and trace_ir._ts(event.get("timestamp")) is not None
                            and a.get("bg_finished_ts")
                            and trace_ir._ts(event["timestamp"]) >= trace_ir._ts(a["bg_finished_ts"])):
                        text = obs.get(event["call_id"]) or ""
                        rows = re.findall(r"(?m)^\s*\d+[→\t](.*)$", text)
                        if rows:
                            a["bg_original_output"] = "\n".join(rows)
    if any(a["tool"] == "unified_exec" for a in plan["actions"]):
        from .agents.codex import resolve_polled_commands
        resolve_polled_commands(plan, ir["steps"][:k])
    return plan


# A file the agent *uses* after the cut: a `Read` call, or a shell command that prints one file's content. Only
# absolute paths are taken (a relative path's meaning depends on a cwd the probe cannot reconstruct reliably), and the
# harness's own scaffolding plus device/pseudo files are skipped.
_POST_CUT_READ = re.compile(r"(?:^|[\s;&|(])(?:cat|head|tail|less|more|tac|nl|wc|zcat|md5sum|sha256sum|stat|file|"
                            r"jq|cut|sort|uniq|diff|od|base64|strings)"
                            r"\s+(?:-\S+(?:\s+\d+)?\s+)*[\"']?(?P<path>/[^\s'\"<>|;&()]+)")
# a glob, a shell variable or a substitution cannot be probed as one file: they are not "the file the agent reads"
_POST_CUT_UNPROBEABLE = ("*", "?", "[", "$", "%")
_POST_CUT_SKIP = ("/tmp/", "/dev/", "/proc/", "/sys/", "/logs/agent", "/var/log/", "/.asr-", "/root/.claude/",
                  "CLAUDE_CONFIG_DIR")


_ABS_TOKEN = re.compile(r"[\"']?(/[^\s\"'<>|;&(),]+)[\"']?")


def absolute_paths_before(ir: dict[str, Any], k: int) -> set[str]:
    """Absolute paths named by the session before the cut (the files pre-cut work produced or touched)."""
    out: set[str] = set()
    for step in ir["steps"][:k]:
        for e in step["events"]:
            if e.get("type") != "tool_call":
                continue
            text = json.dumps(e.get("args") or {}, ensure_ascii=False)
            for m in _ABS_TOKEN.finditer(text):
                out.add(m.group(1).rstrip("."))
    return out


def post_cut_referenced_files(ir: dict[str, Any], k: int, max_paths: int = 120) -> list[dict[str, Any]]:
    """Absolute files the agent reads after step k that pre-cut work had already produced.

    The lenient replay gate requires these to exist in the replayed environment: the resumed agent's history is
    restored verbatim, so a file it already "remembers" reading must still be there when it reads it again or builds
    on it. A path that is only named *after* the cut is the agent's own future output (it does not exist yet by
    design), so it is not required."""
    pre = absolute_paths_before(ir, k)
    seen: dict[str, dict[str, Any]] = {}
    for step in ir["steps"][k:]:
        for e in step["events"]:
            if e.get("type") != "tool_call":
                continue
            tool, args = e.get("tool"), e.get("args") or {}
            found = []
            if tool in ("Read", "NotebookRead"):
                p = args.get("file_path") or args.get("path") or args.get("notebook_path")
                if p:
                    found.append((str(p), tool))
            elif tool == "Bash":
                for m in _POST_CUT_READ.finditer(args.get("command") or ""):
                    found.append((m.group("path"), "Bash"))
            for path, how in found:
                if not path.startswith("/") or any(s in path for s in _POST_CUT_SKIP):
                    continue
                if any(ch in path for ch in _POST_CUT_UNPROBEABLE):
                    continue
                path = path.rstrip(",;:)]")  # trailing punctuation from a shell one-liner
                if path not in pre and path.rstrip(".") not in pre:
                    continue  # produced after the cut: reading it before it exists is expected
                seen.setdefault(path, {"path": path, "how": how, "atif_step_id": step.get("atif_step_id")})
                if len(seen) >= max_paths:
                    return list(seen.values())
    return list(seen.values())


def step_for_cut_line(ir: dict[str, Any], cut_line: int, trimmed=None) -> int:
    """ATIF step id of the agent step that starts at session line cut_line (0-based; = number of prefix lines kept).
    Accepts the step's first line or its trimmed cut (adapter.cut_line, e.g. without trailing bookkeeping lines)."""
    for st in ir["steps"]:
        if st["source"] == "agent" and (st["line_range"][0] == cut_line or
                                        (trimmed is not None and trimmed(st["index"]) == cut_line)):
            if st["atif_step_id"] is None:
                raise SystemExit(f"session line {cut_line} has no aligned ATIF step")
            return st["atif_step_id"]
    near = sorted((abs(st["line_range"][0] - cut_line), st["line_range"][0], st["atif_step_id"])
                  for st in ir["steps"] if st["source"] == "agent")[:4]
    raise SystemExit(f"no agent step starts at session line {cut_line}; nearest (line, step): "
                     f"{[(ln, sid) for _, ln, sid in near]}")


def build(*, adapter, trial_dir: Path, atif_step: int | None, out: Path, mode: str = "same", hint: str | None = None,
          agent_timeout: float | None = None, task_dir: Path | None = None, run_id: str | None = None,
          model: str | None = None, pin_date: bool = True, fingerprint_roots: list[str] | None = None,
          force: bool = False, persist_hint: bool = False, cut_line: int | None = None,
          prior_hints: list[str] | None = None, base_snapshot: dict[str, Any] | None = None,
          seed_carried: bool = True) -> tuple[Path, dict]:
    if mode not in ("same", "hint"):
        raise SystemExit("--mode must be same or hint")
    if mode == "hint" and not hint:
        raise SystemExit("--mode hint needs --hint-file")
    if prior_hints and mode != "hint":
        raise SystemExit("--prior-hint-file needs --mode hint (or none: these bundles keep the chain metadata)")
    trial_dir = Path(trial_dir).resolve()
    trial = trial_dir.name
    result = _trial_result(trial_dir)
    task = (result.get("task_name") or "").split("/")[-1] or (task_dir.name if task_dir else trial.split("__")[0])
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    timeout = resolve_agent_timeout(trial_dir, agent_timeout, task_dir)
    ir = adapter.load_trace(trial_dir, agent_timeout_s=timeout, task=task, reward=reward)
    if ir.get("error"):
        raise SystemExit(f"{trial}: {ir['error']}")
    if (atif_step is None) == (cut_line is None):
        raise SystemExit("give exactly one of --step / --cut-line")
    sess_path = trial_dir / ir["sources"]["session_jsonl"]
    native_lines = None
    if getattr(adapter, "wire", None):
        with sess_path.open("rb") as stream:
            native_lines = stream.readlines()
        raw_lines = [line.decode("utf-8").rstrip("\r\n") for line in native_lines]
    else:
        with open(sess_path, encoding="utf-8") as f:
            raw_lines = [l for l in f.read().split("\n") if l.strip()]
    if atif_step is None:
        atif_step = step_for_cut_line(ir, int(cut_line), lambda i: adapter.cut_line(raw_lines, ir, i))
    k = trace_ir.step_index(ir, atif_step)
    st = ir["steps"][k]
    if st["source"] != "agent":
        raise SystemExit("resume step must be an agent step")
    prev = ir["steps"][k - 1]
    cut = adapter.cut_line(raw_lines, ir, k)
    cut_orig = cut  # the plan and the base snapshot speak about the parent's own session, before any seed normalisation
    plan = plan_with_observations(ir, k)
    # what the agent will read after the cut: the lenient replay gate requires these to survive the replay
    plan["post_cut_paths"] = post_cut_referenced_files(ir, k)
    # ---- chained round-N build: a resumed trial's session must be normalised before it can seed the next run
    evidence = chain.run_evidence(trial_dir)
    if persist_hint and mode != "hint" and not evidence:
        raise SystemExit("--persist-hint needs --mode hint (a chained mode=same bundle may use it too)")
    chain_audit: dict[str, Any] | None = None
    seed_lines = raw_lines
    carried: list[str] = []
    wanted: list[str] = []
    hint_anchors: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    base = None
    if getattr(adapter, "wire", None) and (evidence or persist_hint or prior_hints):
        raise SystemExit("chained resume and persistent hints are supported for Claude Code only")
    if evidence:
        carried = chain.hints_of(evidence["spec"])
        if prior_hints is not None and [h.strip() for h in prior_hints] != carried:
            raise SystemExit("--prior-hint-file does not match the hints this run carried "
                             f"({len(carried)} hint(s) recorded in its proxy spec)")
        if not seed_carried:
            # The round-N hint restates the earlier decisions and the proxy injects it at the cut as one message, so
            # the carried hints are not placed in the history (only their resume scaffolding is removed).
            carried = []
        wanted = carried + ([hint] if mode == "hint" and hint else [])
        if not evidence.get("first_request"):
            raise SystemExit(f"{trial}: no agent/resume-proxy/*/first_request_rewritten.json; cannot place the "
                             "hints the run carried (needed to seed a chained resume)")
        recs = [json.loads(trace_ir._REDACT_RE.sub(r"\1null", l)) for l in raw_lines]
        art = evidence["spec"].get("resume_artifacts") or {}
        try:
            if base_snapshot:
                chain.check_base_snapshot(base_snapshot, trial=trial, raw_lines=raw_lines)
            stripped, strip_audit = chain.strip_scaffolding(recs, evidence["spec"]["sentinel"],
                                                           art.get("continue_text", chain.CONTINUE_TXT),
                                                           art.get("synthetic_assistant_text", chain.SYNTH_TXT))
            cut = chain.cut_after_strip(recs, stripped, cut)
            # The hints this run already carried are NOT seeded into its session. Claude Code drops every copy a
            # seed can carry -- a text block appended to a tool_result record, a user turn of its own, and an edited
            # compact summary all disappear when it rebuilds the request on --resume (measured on 2.1.280). The
            # parent's proxy re-injected them on every request instead, so this run's proxy does the same, at the
            # place the parent had them (chain.carried_hint_positions). The new hint is injected at the cut as for a
            # first-round bundle.
            positions = chain.carried_hint_positions(evidence, stripped, cut, carried)
            seeded, hint_audit = stripped, [{"hint": i, "where": "proxy reinjection", **p}
                                            for i, p in enumerate(positions)]
        except ValueError as exc:  # every chained-build refusal is a user-facing error, never a traceback
            raise SystemExit(f"{trial}: cannot seed a chained resume: {exc}") from exc
        # only a hint the parent held on a tool result has an anchor; one re-injected on the summary does not (the
        # chain gate then requires only the ordered single occurrence)
        hint_anchors = [{"block": p["block"], "tool_use_id": p["tool_use_id"]} for p in positions
                        if not p.get("on_summary")]
        seed_lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in seeded]
        chain_audit = {"parent_run_id": (evidence.get("parent_bundle") or {}).get("run_id"),
                       "parent_trial_dir": str(trial_dir), "carried_hints": len(carried),
                       "hints": len(wanted), "stripped": strip_audit, "materialised": hint_audit,
                       "first_request": evidence.get("first_request_path")}
    elif prior_hints:
        raise SystemExit("--prior-hint-file needs a resumed trial dir (only its own session carries the earlier "
                         "hints; an L0 trial never saw them)")
    if base_snapshot and not evidence:
        raise SystemExit("--base-snapshot needs a resumed trial dir (the tail is split on its own session lines)")
    plan["initial_cwd"] = ir.get("cwd") or next((json.loads(trace_ir._REDACT_RE.sub(r"\1null", l)).get("cwd")
                                for l in raw_lines if '\"cwd\"' in l), None)
    plan["shell_env"] = captured_shell_env(trial_dir)
    if base_snapshot:
        line = int(base_snapshot["session_line"])
        if line >= cut_orig:
            raise SystemExit(f"--base-snapshot session line {line} is not before the cut {cut_orig}")
        full = len(plan["actions"])
        plan["actions"] = chain.tail_actions(plan["actions"], line)
        if not plan["actions"] and full:
            raise SystemExit(f"--base-snapshot at session line {line} already covers every action before the cut")
        base = {k: base_snapshot.get(k) for k in ("snapshot_id", "session_line", "ts", "run_id", "record_path")}
        base["n_actions_from_snapshot"] = full - len(plan["actions"])
        plan["base_snapshot"] = base
    trailer = adapter.bg_output_trailer(ir, raw_lines)
    if trailer and any(a.get("side_effect") == "exec_background" for a in plan["actions"]):
        plan["bg_output_trailer"] = trailer
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id or default_run_id(trial, atif_step, mode))
    bdir = Path(out) / run_id
    if bdir.exists():
        if not force:
            raise SystemExit(f"{bdir} exists (use --force to overwrite)")
        shutil.rmtree(bdir)
    (bdir / "aux").mkdir(parents=True)
    trunc = (b"".join(native_lines[:cut]).decode() if native_lines is not None
             else "\n".join(seed_lines[:cut]) + "\n")
    (bdir / "session_trunc.jsonl").write_bytes(trunc.encode())
    with open(bdir / "plan.json", "w") as f:
        json.dump(plan, f, ensure_ascii=False)
    aux_counts = adapter.collect_aux(sess_path, ir, k, trunc, bdir / "aux")
    rp = next(r for r in trace_ir.resume_points(ir) if r["index"] == k)
    budget = round(timeout - (st["elapsed_s"] or 0), 1)
    budget_source = "timeout_minus_session_elapsed"
    if evidence:
        elapsed_run = None
        if (evidence.get("run_start") or {}).get("t") and st.get("timestamp"):
            elapsed_run = chain.ts_seconds(st["timestamp"]) - float(evidence["run_start"]["t"])
        budget, budget_source = chain.remaining_budget(evidence, elapsed_run, budget)
    if budget <= 0:
        raise SystemExit(f"no agent budget left at step {atif_step} (elapsed {st['elapsed_s']} s >= {timeout} s)")
    orig_date = (ir["start_ts"] or "")[:10]
    sentinel = sentinel_for(run_id)
    n_compactions = sum(1 for s in ir["steps"][:k] for e in s["events"] if e["type"] == "compaction_boundary")
    if getattr(adapter, "wire", None):
        orig_request, error = adapter.native_reference(trial_dir, ir, k, seed_lines[:cut])
        loc = {"error": error}
    else:
        loc = parity.locate_request(ir, k, parity.original_requests(trial_dir))
        orig_request = (parity.expected_from_record(loc["record"], index=loc["index"], method=loc["method"])
                        if "record" in loc else None)
        if orig_request is not None:
            if evidence:
                prefix = [json.loads(l) for l in seed_lines[:cut]]
            else:
                prefix = [json.loads(trace_ir._REDACT_RE.sub(r"\1null", l)) for l in raw_lines[:cut]]
            orig_request["content"] = parity.content_reference(
                prefix, adapter.rendered_attachment_templates(ir.get("cc_version")))
    agent_model = model or ((result.get("config") or {}).get("agent") or {}).get("model_name") or ir.get("model")
    roots = fingerprint_roots or default_fingerprint_roots(plan)
    optional_roots: list[str] = []
    if roots is None and not plan["actions"]:
        # An empty replay plan cannot change the task files, and a task image may have no /app at all: fingerprint
        # /app as an *optional* root, so the clone must match "all files present with these hashes" or "absent too".
        from .replay.fingerprint import DEFAULT_ROOTS
        roots, optional_roots = [str(r) for r in DEFAULT_ROOTS], [str(r) for r in DEFAULT_ROOTS]
    meta = {
        "run_id": run_id, "agent": adapter.name, "trial": trial, "trial_dir": str(trial_dir), "task": task,
        "task_path": str(task_dir.resolve()) if task_dir else None, "orig_reward": reward,
        "session_id": ir["session_id"], "project_slug": ir["project_slug"], "cwd": plan["initial_cwd"],
        "agent_version_orig": ir.get("cc_version"), "resume_atif_step": atif_step, "resume_index": k,
        "prev_atif_step": prev["atif_step_id"], "cut_line": cut, "n_session_lines": len(raw_lines),
        "session_sha256": ir["sources"]["session_sha256"], "elapsed_at_k_s": st["elapsed_s"],
        "agent_timeout_task_s": timeout, "agent_budget_s": budget, "agent_budget_source": budget_source,
        "orig_date": orig_date, "mode": mode, "hint": hint, "prior_hints": wanted[:-1] if mode == "hint" else wanted,
        "n_hints": len(wanted), "model": agent_model, "sentinel": sentinel, "aux_counts": aux_counts,
        "resume_cost": rp, "n_replay_actions": len(plan["actions"]), "flags": ir["flags"],
        "n_post_cut_paths": len(plan["post_cut_paths"]),
        "n_compactions_before_cut": n_compactions, "orig_request_at_k": orig_request,
        "orig_request_locate_error": loc.get("error"),
        "fingerprint_roots": roots,
        "fingerprint_optional_roots": optional_roots,
        "replay_reviewed_exceptions": {},
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "orig_step_k_preview": [{"type": e["type"], "tool": e.get("tool"),
                                 "text": (e.get("text") or json.dumps(e.get("args", ""), ensure_ascii=False))[:400]}
                                for e in st["events"] if e["type"] in ("assistant_reasoning", "assistant_text",
                                                                       "tool_call")],
    }
    if getattr(adapter, "wire", None):
        meta.update(prefix_sha256=hashlib.sha256(trunc.encode()).hexdigest(),
                    native_provider=st.get("native_provider") or ir.get("provider"), native_model=st.get("native_model") or ir.get("model"),
                    native_effort=st.get("native_effort"),
                    session_relpath=ir["sources"]["session_jsonl"])
    if st.get("native_sandbox_mode"):
        meta["native_sandbox_mode"] = st["native_sandbox_mode"]
    if chain_audit is not None:
        meta["chain"] = chain_audit
        meta["parent_run_id"] = chain_audit.get("parent_run_id")
    if base is not None:
        meta["base_snapshot"] = base
    with open(bdir / "bundle.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    spec = {"mode": mode, "sentinel": sentinel, "hint": hint if mode == "hint" else None,
            "upstream": "http://127.0.0.1:4013", "model": None, "pin_date": orig_date if pin_date else None,
            "resume_artifacts": adapter.bundle_artifacts(meta["cwd"], meta["orig_request_at_k"])}
    if getattr(adapter, "wire", None):
        spec["wire"] = adapter.wire
        if meta.get("native_sandbox_mode"):
            spec["resume_artifacts"]["sandbox_mode"] = meta["native_sandbox_mode"]
    carried_hints = list(wanted[:-1] if mode == "hint" else wanted)
    if carried_hints:
        spec["prior_hints"] = carried_hints  # both modes: the next round must know which hints this run carried
    if wanted:
        # Every hint the run carries must be in the request exactly once, in this order, and (anchor entries) right
        # after the tool result it originally followed: a lost, duplicated or moved hint fails closed.
        spec["required_hint_blocks"] = [chain.hint_block(h) for h in wanted]
        if positions:
            # Where the previous run's proxy held the hints it carried: this run's proxy re-injects them there on
            # every request (the resumed history itself cannot carry them, see bundle.build).
            spec["prior_hint_positions"] = [dict(p) for p in positions]
        if hint_anchors:
            spec["hint_anchors"] = hint_anchors
        if evidence:
            spec["forbidden_sentinels"] = [evidence["spec"]["sentinel"]]
    if persist_hint:
        spec["persist_hint"] = True  # re-inject every hint on the compaction summary once the sentinel is gone
    with open(bdir / "proxy_spec.json", "w") as f:
        json.dump(spec, f, indent=1)
    return bdir, meta
