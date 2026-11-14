# Design

This document records the design decisions and the facts they depend on. Facts about Claude Code are for version
2.1.207; recheck them before using another version.

## 1. Data model

### Trajectory IR (`trace/`)

`trace/claude_code.py` turns a Harbor trial dir into a plain-dict IR. It uses only the standard library.
`trace/ir.py` works only on that IR and does not depend on any particular agent.

- **Source of truth**: the native session `agent/sessions/projects/<slug>/<sid>.jsonl`. The slug is the agent's
  working directory with `/` replaced by `-` (e.g. `-app`). Subagent sidechains are in
  `<sid>/subagents/agent-*.jsonl`. Large outputs that the agent may `Read` later are in `<sid>/tool-results/*`.
- **Step grouping**: a step starts at the first session line of a new assistant `message.id`. Claude Code writes one
  line per content block. A user line that carries text also starts a step. Tool results, attachments, and system
  lines attach to the current step. `line_range = [start, end)`.
- **Step ids**: if Harbor's ATIF `agent/trajectory.json` is present and its main-chain steps align 1:1 with the IR
  steps (same timestamps), each IR step carries the ATIF `step_id`. Without ATIF, step ids are 1-based positions
  (flag `atif_missing_step_ids_positional`).
- **Events**: `instruction`, `user_text`, `assistant_reasoning`, `assistant_text`, `tool_call` (tool, exact args,
  side-effect class, cwd, duration, is_error, background ids), `observation` (full text, is_error, persisted path,
  background task id), `attachment`, `system`, `compaction_boundary`, `sidechain`.
- **Side-effect classes**: `read_only` (Read/Glob/Grep, task reads, and Bash commands that pass a conservative
  read-only parser), `file_write` (Write/Edit/MultiEdit/NotebookEdit), `exec` (other Bash), `exec_background`
  (`run_in_background`, or auto-backgrounded by Claude Code), `subagent`, `task_mgmt`, `web`, `other`.
- **Redaction**: some pipelines write a bare `[REDACTED]` token as a JSON value, which makes the line invalid JSON.
  The loader parses such lines with `null` in place of the token and adds a flag. The raw bytes are kept unchanged
  for truncation, but such a prefix is no longer faithful.

### Bundle (`bundle.py`)

| File | Content |
|---|---|
| `session_trunc.jsonl` | `lines[:cut]` of the original session, **byte-exact**. `cut` is the first line of step *k*, minus trailing `last-prompt` / `queue-operation` bookkeeping lines, so the file ends with step *k-1*'s tool results and attachments |
| `plan.json` | `replay_plan(ir, k)`: side-effecting actions before *k* in timestamp order (main chain plus sidechain calls whose results landed before *k*), each with its original observation text |
| `aux/tool-results/*` | persisted tool outputs referenced in the prefix |
| `aux/tasks/*.json` | task-list state rebuilt from successful TaskCreate/TaskUpdate calls before *k* |
| `proxy_spec.json` | mode, sentinel, hint, `pin_date`, resume artifacts (no secrets) |
| `bundle.json` | run id, session id, slug, cwd, `cut_line`, `agent_budget_s`, sentinel, resume cost, flags, optional `fingerprint_roots`, `replay_reviewed_exceptions` |

**Budget**: `agent_budget_s = agent_timeout − elapsed(k)`. `agent_timeout` is the task's `[agent].timeout_sec`
(or `--agent-timeout`). `elapsed(k)` is measured from the first step's timestamp. The resumed agent gets this as
Harbor's `override_timeout_sec`. The sandbox also enforces it itself (`bounded_remote_command`: `timeout -s TERM
-k 10s`), so the limit holds even if the host controller dies.

## 2. Environment: replay → gate → snapshot → clone

### Source preparation (`sandbox/source.py`)

1. Create a source sandbox from the task's snapshot-capable template. Run the task's healthcheck and
   `chmod 1777 /tmp` (the FC service creates `/tmp` as 0777; some verifiers check the sticky bit).
2. Install the pinned agent: Claude Code 2.1.207 official linux-x64 binary, downloaded and checked with
   `sha256sum -c` against the release manifest checksum. Install the proxy runtime: python-build-standalone CPython
   3.12.12 (release asset digest pinned) plus `aiohttp` and its dependencies from PyPI with `--require-hashes`.
   No key is involved and no proxy is started.
3. Upload `plan.json` and `replay/replay_inside.py`, and **dispatch the replay exactly once** (SDK background
   command). It reports completion through an atomically renamed `rc` file. The replay is never re-dispatched:
   replayed commands are not idempotent.
4. Run the **replay gate**, check auxiliary-file hashes, delete the replay scaffolding and `/logs/*`, and record
   the fingerprint of the task roots.
5. `create_snapshot()`, record its id, and kill the source. The snapshot contains no secret.

### Replay policy (`replay/replay_inside.py`, stdlib-only)

- `Write`: write the content exactly (create parent dirs). `Edit`: apply exact `old_string → new_string`
  (`replace_all` respected). An edit that failed originally must fail again (`error_reproduced`).
- Foreground `Bash`: run in the original cwd with timeout `max(60, 2×orig_duration+30)`, capped by `--exec-cap`. If
  the original timed out, use the original timeout. Kill the whole process group on timeout.
- Background `Bash` that finished before *k*: re-run in the foreground. Its output file is restored and hashed.
  Background `Bash` still running at *k*: start it detached and flag it.
- Subagents: no model call. Their own side-effecting tool calls that finished before *k* are replayed in timestamp
  order.
- Task list: restored from the session by the adapter, not by the replayer. Web calls: not replayed. Read-only
  calls: skipped.
- Output comparison classes: `exact`, `normalized` (whitespace/ANSI), `same_lines_reordered` (stdout/stderr
  interleaving), `volatile_only` (addresses, dates, timings), `address_and_order_only`,
  `line_alignment_and_order_only`, `numbers_differ`, `differs`. The gate accepts the three order classes only when
  the reordering is explained by stderr placement (the original equals the replay's stdout in order once the
  replay's stderr lines are taken out). `volatile_only` is not accepted as such: a line may differ only as an
  `ls -l` row whose date column alone changed and whose original date is at/after the original run's first action
  (a file the run created; input files keep their dates), or, in a timing context (the command's code, or a script
  it runs, measures time — the replayer records `scripts_measure_time` inside the sandbox; the shell `time`
  keyword only covers its own report), in durations in a measurement form (after a timing label, in parentheses,
  after `OK`, or ending a `<words>: <duration>` line; never `key: value` / `key=value`) within run-to-run jitter.
  The grep no-match class requires the last command to be a plain pipeline ending in grep (no `&&`/`||`/`&`,
  stderr redirection, subshell or pipefail). Addresses, pids, timestamps and other data are never masked
  automatically. An empty successful result equals Claude Code's
  `(Bash completed with no output)`.
- Background output files get the agent's completion trailer (`plan.json["bg_output_trailer"]`; Claude Code 2.1.280
  appends `\n[exited with code N]\n`), so later reads of the file see the same bytes.
- Pacing: a replayed foreground command that leaves processes running (`nohup … &`) keeps its process group alive.
  While it is alive, each later `Bash` action waits until the original time gap since that command has elapsed
  (bounded by `--exec-cap`; recorded as `paced_wait_s`), so polls and log reads see a comparable progress point.
  If the gap cannot be kept (beyond the cap, or a start time is unknown) the action is `pacing_unmet` and the gate
  fails. Replay otherwise runs back to back.

### Replay gate

`replay/gate.py::validate_replay` fails closed. It requires that every planned action ran, in order with the same
call ids, and that every result has an accepted fidelity:

- accepted: `exact`, `normalized`, `same_lines_reordered`, `volatile_only`, `address_and_order_only`,
  `line_alignment_and_order_only`, `error_reproduced`, `orig_error_not_applied`;
- background jobs: same terminal status as the original (`completed` → rc 0, `failed` → rc ≠ 0), output file
  restored with a hash;
- the exit status of an `exec` (error or not) must match the original;
- **automatic equivalence** (no review; each class narrow and checked on the full output, `automatic_equivalence`
  in the report): `ls_dot_entries_only`, `cc_grep_no_match` (last pipeline stage grep returns 1, Claude Code showed a
  non-error, stdout identical), `readonly_listing_order_only` (command without writes lists a directory; each
  output line has the same tokens), `shell_diagnostic_prefix_only` (`/bin/bash: eval: line N:` vs
  `bash: -c: line M:`), `shell_time_report_only` (`time`/`TIMEFORMAT`/`/usr/bin/time` report lines removed, rest
  identical after volatile masking), `cprofile_timings_only` (same profile rows and call counts; timing columns
  differ), `ls_mtime_and_stderr_placement_only` (command runs `ls -l`; only the date column of ls rows differs,
  plus stderr placement), `timing_lines_only` (command measures time; every differing number directly follows a
  duration label or precedes a time unit). Stderr placement means: the original equals the replay's stdout in
  order once the replay's stderr lines are taken out of it; nothing else is reordered;
- pacing: an action marked `pacing_unmet` (a background job was alive and the original gap could not be kept)
  fails the gate;
- `numbers_differ` / `differs` are accepted only with a **reviewed exception** in
  `bundle.json["replay_reviewed_exceptions"][call_id]`, bound to `sha256(command)`. Supported kinds:
  `hardware_probe`, `hardware_probe_and_python_build_stamp`, `shell_memory_probe`, `terminal_grep_no_match`,
  `terminal_grep_with_shell_time`, `original_timeout_partial_output`, `cprofile_timings`,
  `cprofile_reviewed_aggregate_counts`, `ls_dot_entries` (`ls -l` whose `.`/`..` rows and `total` differ because
  the source filesystem lists them and the original did not; all other rows equal field by field),
  `unseeded_rng_numbers` (read-only probe using an unseeded RNG; same text with numbers masked),
  `unlabelled_timing_number` (command measures time; exactly one number differs), `listing_order_line` (same
  items listed in another order; the reviewer checks nothing depends on that order), `agent_state_read` (the
  original read the agent's own session/log files, which exist only from launch on; the replay must fail with
  "No such file" on such a path and print nothing; the reviewer checks the command writes nothing),
  `reviewed_masked_output` (the reviewer compared both outputs and they carry the same information once
  `gate.mask_volatile` removed machine spec — `Mem:`/`df`/`CPU(s)`/host model — dates, UUIDs, hex object
  addresses, `ls` block counts and `./..` rows, and numbers; `order` is `ordered`, `sorted` or `subsequence`
  (a listing cut short by `head`), the last two only for `ls/find/du/tree/stat/wc`, and the exception must
  record `changed_numbers_reused_later: false` — i.e. no changed number reappears in the agent's later
  reasoning or tool arguments; a timeout or a changed exit outcome is never masked away) and
  `web_readonly_not_replayed` (a `WebFetch`/`WebSearch` in the prefix, an external read the gate cannot re-issue;
  bound to `sha256` of the tool input, read-only, no local state). Each kind checks that only that specific benign difference exists. Record a `note` next to
  each exception.

### Screening cuts offline (`asr screen`)

`asr screen --trial-dir T --cut-line L` (repeatable, or `--step`) builds the replay plan of each candidate cut and
lists what may make the replay diverge, without a sandbox: `bg_inflight` / `subagent_inflight` (block), reads of a
background job's output while it was still running (block) or after it finished (info), shell background jobs and
polls (`pace_dependent_read`), `unseeded_rng`, `network`, `glob_order`, `hash_order`, `agent_state_read`,
`web_or_unsupported`, `wall_clock` / `orig_timeout` / `long_replay` / `compactions` (info) and `missing_original`
(the original request at the cut was not located, so the parity gate cannot run). Verdict: `block`, `review` (warns
only) or `ok`. A warn is not a failure: the replay gate decides; the screen tells you where to look.

### Clone (`sandbox/snapshot_env.py`)

`SnapshotCloneEnvironment` creates the agent sandbox (Harbor session id ending in `__env`) from the snapshot. The
verifier sandbox is created normally from its own template. Before the agent starts, these checks must pass:

- `expected_fingerprint`: the recorded fingerprint command is run in the clone, and its output must equal the
  source's;
- `expected_auxiliary_hashes`: restored background outputs;
- `expected_verifier_hashes`: the verifier sandbox's `/tests/*` must match the local task's `tests/`.

Results are written to `<trial>/infra-validation/`. FC clones reject create-time env vars, so none are passed.

### Command transport recovery (`sandbox/command_recovery.py`)

The resumed agent runs as one long command. If the RPC stream breaks, the command is **not** re-dispatched:

- stdout, stderr, and the exit code go to a private remote directory (a durable receipt);
- on a transport error (SSL/connection), reconnect to the **same PID** within the caller's deadline; the deadline is
  never extended;
- an unacknowledged dispatch is never retried, and cancellation does not restart the command;
- every step is journaled to `<trial>/command-recovery/<session>/<token>.json`.

## 3. Context: resume + rewrite

### What `claude --resume <sid> --print <prompt>` adds (2.1.207)

1. a text block `Continue from where you left off.` appended to the last tool-result user message;
2. a `<synthetic>` assistant message `No response requested.`;
3. the user prompt. We pass a unique **sentinel** (`ZZZ_RESUME_SENTINEL_<sha1(run_id)[:10]>`);
4. sometimes resume-time notes after the prompt, e.g. `<file> was modified ...` when the replay changed a file
   Claude Code had cached.

Claude Code writes these into the session, so they appear again in **every** later request.

2.1.280 adds the same 1–3 and, after the sentinel, a system message (`# Environment … OS Version … <total_tokens>
… The date has changed. Today's date is now <today>`) that carries the request's trailing prompt-cache breakpoint.
The proxy drops it (post-sentinel rule, step 3 below) and moves the breakpoint to the new last message.

### Cuts after context compactions

A session with `compact_boundary` system lines is rebuilt by `claude --resume` (2.1.280, checked against real
sessions with a fake upstream) exactly as the original run's request after the compaction: the compact summary
user message (a plain string), the persisted date reminder as a system message, the preserved messages
(`compactMetadata.preservedMessages`), the later turns, with attachments rendered as system messages or as
`<system-reminder>` suffixes of tool results. Two things needed care:

- `pin_date` must not reshape the first message: after a compaction it is the summary, sent as a string; turning it
  into a block list adds bytes. The proxy keeps the content's shape and never edits a compaction summary.
- `asr equiv` rebuilds the reference context the same way (`equiv.context_lines`: after the last boundary, the
  summary, then the preserved messages, then later lines) and accepts `<system-reminder>` text appended to a
  tool_result's string content (counted against the prefix's attachments).

### First-request parity gate

`build-bundle` locates the original run's request that produced step *k* in the original passthrough proxy log
(`agent/model-proxy/requests-*.jsonl`) by step *k*'s token usage, which is unique per request; counting assistant
messages is wrong when failed attempts or compactions were logged. It records `orig_request_at_k` (index,
`req_bytes`, model, n_messages, max_tokens, thinking, n_tools) in `bundle.json`. At launch the agent puts it into the
resume proxy spec (`first_request_parity`, with the request transformations of `--passthrough-args`). The proxy
rejects the first request with 400, before anything is sent upstream, unless

- the rewritten body minus the injected hint block(s), serialised as the original route logged it, has exactly the
  original `req_bytes`, and model, n_messages, max_tokens, thinking and tool count are equal; and
- its model-visible content matches the session prefix (`content`, built by `parity.content_reference`): the
  system prompt equals the agent's last persisted `prompt_snapshot` (2.1.280 writes it into the session), every
  recorded tool has the same name, description and input schema (only the deferred-loading placeholder may be
  added), and every message block of the rebuilt context is present in order and byte-identical; other blocks may
  only be system-role messages, the hint block, or user text that starts with a rendered-attachment marker, no
  more of them than the prefix has attachments, and every line of them must be explained by the attachments
  persisted at the same context position (block index, +-1 for a reminder rendered onto the neighbouring tool
  result), parsed in order: a whole persisted leaf value of one of them as an exact contiguous block (multi-line
  file contents verbatim or numbered as the Read tool renders them; each leaf used once, so reordered, duplicated or
  omitted lines fail), a fixed rendering line, or a slot template of the agent version
  (`CC_RENDERED_ATTACHMENT_TEMPLATES`, collected from real 2.1.280 resumes) whose every slot is filled by the named
  field (`<attachment type>.<path>`) of one of those attachments — so no instruction text, date, token count or
  environment value can appear that the original did not carry there, and values cannot move between fields or
  positions. An agent version without
  templates cannot bind rendered text and fails. With `--parity-gate require` a prefix without a prompt snapshot
  (system prompt and tools unbound) fails.

Byte length alone would accept an equal-length substitution; the content check closes that for the persisted
history. Rendered attachment text and request settings that are not in the original proxy log (e.g. sampling
parameters, the prose of a reminder) cannot be bound to the original run, so the strict mode adds a second, exact
gate for real runs: `launch --freeze-first-request <first_request_rewritten.json>` takes the first request of a
`--fake` run of the same bundle that passed the parity gate and requires the real run's first request to have the
same digest (every field except `metadata`; `expected_first_request_sha256`). Provenance is verified: the fake
run's parity report must have judged exactly that request (`request_sha256`), come from a fake run, with the same
passthrough route as this launch, and the fake run's `bundle_identity` (replay plan and reviews, fingerprint
roots, cut, session seed and auxiliary files, proxy spec, original request; written by the agent at startup) must
equal this bundle's. The agent re-runs the same verification at startup from `freeze_first_request` (a bare digest
in a hand-written job config is refused). Launch also always checks the snapshot's cut identity against the
bundle, even when the run id is the same. Fake and real runs of one bundle send
identical first requests (checked on two real runs), so this binds everything a fake run can show. With
`--parity-gate require` a real launch without `--freeze-first-request` is refused. Once a gate failed, every later
request of that run is refused too. The report is `first_request_parity.json`.
`--parity-gate auto` (default) gates whenever the bundle has the record, `require` refuses to launch without it,
`off` disables it. `asr parity <bundle> <first_request_rewritten.json>` runs the same check offline.

### Sharing one snapshot between bundles

The snapshot depends only on the cut, not on the mode or hint. `launch` accepts a snapshot state prepared from
another bundle only if `bundle.snapshot_cut_identity()` is equal for both bundles: same trial, session sha256,
cut line, ATIF step, agent and version, fingerprint roots, plan sha256 and reviewed exceptions. So build the `same`
and `hint` bundles of a trial at the same *k* (with the same `--fingerprint-root`s and the same reviewed exceptions),
prepare one snapshot from either, and launch both from it.

### Resume proxy (`proxy/resume_proxy.py`, port 4031)

On every `/v1/messages` request it:

1. finds the user message that contains the sentinel and drops it;
2. drops the synthetic assistant message before it;
3. drops the notes added after the sentinel turn (they are logged);
4. drops the exact `Continue from where you left off.` block from the preceding user message, skipping past any
   historical `system` messages in between;
5. in `mode=hint`, appends exactly one `<system-reminder>\n<hint>\n</system-reminder>` block to that message;
6. rewrites `Today's date is YYYY-MM-DD.` to the original run date (`pin_date`).

Requests without the sentinel, such as subagent calls, pass through unchanged (`sentinel_not_found`). Optional gates
in the spec: `require_sentinel`, `require_text_only`, `expected_first_request_sha256` (the first request must hash
to a known value, `metadata` excluded, or it is rejected with 400 and never sent upstream), `first_request_parity`
(above), `max_model_requests` (refuse request N+1, for bounded probes), and `required_hint_blocks` /
`hint_anchors` / `forbidden_sentinels` for chained resumes.

The resume artifact texts come from the adapter (`proxy_spec.json["resume_artifacts"]`), so other agents can reuse
the proxy. `upstream: "fake"` returns a canned `end_turn` reply (dry run).

### Passthrough proxy (`proxy/passthrough_proxy.py`, port 4013)

It forwards the request body byte-for-byte to `ASR_UPSTREAM_URL`, removes the incoming auth headers, and injects
the real key. It retries connection errors that happen before any byte was sent. It logs one JSONL record per
request (status, latency, time to first byte, sizes, thinking-block statistics). Optional compatibility flags
(`--drop-fields`, `--strip-is-error`, `--strip-thinking`) exist for upstreams that reject some fields. They are off
by default; turning one on changes the request.

### Key handling

- The agent gets `ANTHROPIC_API_KEY=dummy-resume` and `ANTHROPIC_BASE_URL=http://127.0.0.1:4031/r/<run_id>`.
- `ASR_UPSTREAM_KEY` is read from the Harbor process env and sent as the env of **one** root exec
  (`ASR_PROXY_KEY`). Harbor redacts env names that contain `KEY` in its logs.
- That exec writes a restart-loop script that contains no secret, starts it detached, and unsets the variable.
  The loop copies the key into a non-exported shell variable and gives it to the proxy as
  `--key-file <(printf '%s' "$KV")` (a pipe).
- Result: the key is never written to disk, never appears on a command line, and is not in the proxy process
  environment. It is not in the snapshot, the bundle, or the job config. `tests/test_proxy_credentials.py` runs the
  generated shell and checks all of this.

## 4. Version pins

| Item | Pin | Why |
|---|---|---|
| Harbor | `harbor[e2b]==0.23.0` | template names are derived from task dir contents using version-specific rules |
| e2b SDK | `2.32.0` | tested against the FC sandbox API |
| Claude Code | `2.1.207` (sha256 `85e7e988…5c83`) | must match the original run. 2.1.280 resumes a 2.1.207 session but has a different tool list (no TaskCreate/TaskUpdate/TaskList, adds ListAgents) and adds an attribution reminder |
| Proxy runtime | CPython 3.12.12 (python-build-standalone 20251014, digest pinned) + hash-pinned aiohttp 3.14.5 wheels | reproducible, and independent of the task image's Python |
