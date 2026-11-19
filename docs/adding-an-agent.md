# Adding an agent adapter

An adapter connects one agent CLI (Claude Code, Codex, ...) to the agent-independent machinery: the bundle builder,
replay gate, snapshot clone, proxies, and equivalence check. It is one module, `src/agent_snapshot_resume/agents/<agent>.py`,
plus one line in `agents/__init__.py`.

Use `agents/claude_code.py` as the reference implementation.

## 1. Answer these questions about the agent first

| Question | Claude Code 2.1.207 answer |
|---|---|
| Where does it persist the conversation, and in what format? | `$CLAUDE_CONFIG_DIR/projects/<slug>/<sid>.jsonl`, one line per content block |
| Can it resume a persisted session non-interactively? With which command? | `claude --resume <sid> --print <prompt>` |
| What does resuming add to the model context? | `Continue from where you left off.` block, synthetic `No response requested.` assistant, the prompt, sometimes post-prompt notes |
| Does it re-send those additions on every request? | yes (persisted into the session) |
| Which wire API does it use, and how do you point it at a proxy? | Anthropic `/v1/messages`; `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` |
| Which files besides the session must exist for a faithful resume? | `<sid>/tool-results/*`, `tasks/<sid>/*.json` |
| Is there an official release artifact with a published sha256? | yes, `downloads.claude.ai/claude-code-releases/<v>/manifest.json` |
| Does a newer version change the tools or system prompt when resuming an old session? | yes, so the version must be pinned |

### Native CLI answers

| Question | Codex CLI 0.154.0 | Pi 1.4.0 |
|---|---|---|
| Session | Harbor exports `agent/sessions/YYYY/MM/DD/rollout-*.jsonl`. A response may contain reasoning, commentary and multiple parallel calls. The token-count event closes that API step. | `agent/pi/sessions/*.jsonl`, v3 tree. Walk parent IDs back from the last entry. Each assistant message is one model response. |
| Command | `codex exec resume --json --skip-git-repo-check <session-id> <sentinel>` with a private `CODEX_HOME` containing the dated tree. Six recorded HTTP requests exercised this command. | `pi --session <seed.jsonl> --provider <provider> --model <id> --mode json --print -- <sentinel>`. Harbor also offers `--continue` for the most recent session. |
| Resume additions | The stock CLI inserts a developer message containing skills and permissions, a user environment message, then the sentinel. The proxy validates their adjacency, exact content templates and expected runtime paths. It rejects unrecognized templates. | A single user sentinel message in the two measured requests. The native seed restores the system checkpoint, previous calls and tool results. |
| Wire | OpenAI Responses, POST `/v1/responses`. Set the named provider's `base_url` and `wire_api = "responses"` in `config.toml`. | Tested with `models.json` provider API `openai-completions`, POST `/v1/chat/completions`. The API is recorded on assistant entries. Other Pi providers need their own validation. |
| Cut and auxiliary state | Keep through the preceding calls, outputs, token accounting and any explicit `compacted.replacement_history`. Exclude the new assistant's mirrored completion events. Bundle the required model catalog from `agent/model-catalog.json` or the local `.cache/deepseek-models.json`. The adapter verifies its pinned SHA-256 at build and resume, then seeds `model-catalog.json` and selects it via `model_catalog_json`. A unified-exec session ID refers to process state. | Truncate immediately after the target assistant's parent. A compaction entry includes the summary and first retained entry. Built-in fixture tools need only the session, matching cwd and matching provider/tool flags. Branches with abandoned side effects require separate workspace evidence. |
| Replay mapping | Shell and foreground unified exec → Bash. Custom apply_patch → Bash invoking the pinned native patch engine. A completed command followed solely by empty polls replays once with its combined output. Interactive stdin, interleaved processes and a still-live process at the cut block strict replay. | read → Read, bash → Bash, write → Write, edit → Edit or simultaneous MultiEdit. Unknown tools remain visible in the plan. |
| Installation pin | Linux x64 npm platform archive SHA-256. The local Darwin probe requires both npm archives and verifies each against its registry SHA-512. | SHA-256 over the distributed paths and contents of the installed 1.4.0 package, checked before npm installs it. The Linux installer verifies Node 22.23.4 by archive SHA-256. npm resolves transitive dependencies during installation. |
| Evidence at the cut | Original digests bind to session and completed response ID. For older logs, unique usage binds to metadata and the rollout supplies the projected input. For `deepseek-flash`, the catalog sets 10,000 tokens. Codex 0.154.0 grants 120% for history, estimates four UTF-8 bytes per token and retains up to 24,000 bytes at each end of an oversized string. An equal partial reference gives `incomplete_evidence`. A known difference gives `mismatch`. The strict default requires complete evidence. The limited historical exception is described below. | The baseline recorder joins completed upstream response IDs to native assistant entries, binding raw-body SHA-256, per-message and per-field digests. Two native resumes matched the actual forwarded bytes. |

### Native batch runs

A control-only manifest needs the columns `task,cut_line,source_dir`. Use the adapter name in both phases.

```sh
asr batch prep --name demo --agent pi --manifest manifest.csv --tasks-dir tasks --hinted-reps 0 --control-reps 1
asr batch launch --name demo --agent pi --manifest manifest.csv --tasks-dir tasks --hinted-reps 0 --control-reps 1
asr batch status --name demo --agent pi --manifest manifest.csv --json
```

For a fresh Pi trial, use Harbor agent import path `agent_snapshot_resume.agents.pi:PiBaseline`,
agent kwarg `version=1.4.0` and model name `asr-reasoning/<model-id>`. This provider enables
DeepSeek-compatible thinking, requests maximum effort and preserves assistant reasoning content.
Harbor applies the task timeout. Configure the upstream origin and credential in the Harbor host
environment and grant the model host through Harbor's agent-phase allowlist. One completed Chat
response ID must match exactly one assistant's `responseId`. Truncated streams cannot supply receipts.
Use the resulting trial directory as the manifest source.

For Codex, pass `--agent codex`. A custom translator takes `--passthrough-script PATH` and
`--passthrough-script-sha256 SHA256` on launch or batch. The required script CLI accepts
`--host --port --key-file --upstream --cafile --log-dir --proxy` and serves `/_health`.
The proxy passes an empty outbound proxy setting and supplies the credential through a pipe.
The host and guest verify the script before it runs. Use the identical script and options for both phases.

`--allow-incomplete-evidence` with `--parity-gate require` (implicit in batch) admits only a
fully matched historical rollout prefix whose sole documented gap is the original request envelope.
When historical translation metrics exist, the pinned script must also expose
`build_chat_request(body) -> (payload, auxiliary, metrics)`. The gate verifies each recorded metric.
The report retains `verdict=incomplete_evidence`, `ok=false` and `admitted=true`.
The real run repeats the check and must match its fake run's frozen digest. For a translator that
demonstrably drops `client_metadata`, that digest binds the remaining Responses fields and the entire
translated payload. The capture retains the full Responses body and its digest.

The Pi exporter can record the CLI version in `agent/pi/version.txt` or Harbor's
`result.json.agent_info.version`. For either native adapter, a recorder can write
`agent/model-proxy/native-reference.jsonl` with one object per original response.
Use `session_id`, `entry_id` (Pi entry ID or Codex response ID), and `content` from
`proxy.native_wire.body_reference(wire, original_body)`. Add `raw_sha256` to
`content` for a byte-level check on the compact JSON sent by the proxy. These
records contain digests. Capture each original body before generating its response,
bind it to the entry created by that response, and reject ambiguous joins.
See [the scripted Pi upstream and recorder](../examples/offline/pi_probe.py).

For original Codex runs, enable `--record-request-digests` on the Responses
passthrough proxy. The default leaves digest capture off. Each opted-in request
writes ordered item hashes and their type, role and call ID where present, all
top-level field hashes (including tools and instructions), and the raw-body SHA-256
to `agent/model-proxy/requests-YYYYMMDD.jsonl`. The recorder joins only successful,
completed responses with an identifiable thread. It supports JSON and streamed
Responses events. A complete reference also checks generated IDs, client metadata
and compact serialized bytes. Fresh IDs or different envelope fields can therefore
produce a genuine mismatch. Earlier count-only logs cannot provide these hashes.

If the agent cannot resume a session non-interactively, this approach does not work for it without patching the
agent.

## 2. Host side: subclass `AgentAdapter`

```python
from agent_snapshot_resume.agents.base import AgentAdapter

class CodexAdapter(AgentAdapter):
    name = "codex"
    version = "<pinned version>"
    binary_sha256 = {"<pinned version>": "<sha256 of the linux-x64 artifact>"}
    resume_artifacts = {"continue_text": "<exact text the agent injects>",
                        "synthetic_assistant_text": "<exact text, if any>"}
    harbor_import_path = "agent_snapshot_resume.agents.codex:CodexSnapshotResume"
    default_env = {...}            # non-secret env for the resumed process

    def find_session(self, trial_dir): ...
    def load_trace(self, trial_dir, agent_timeout_s, task=None, reward=None): ...   # -> IR
    def cut_line(self, raw_lines, ir, k_index): ...
    def collect_aux(self, session_path, ir, k_index, prefix_text, out_dir): ...      # optional
    def install_commands(self, version=None): ...    # [(name, shell, timeout_s)], must sha256-check
    def sanity_check_bundle(self, bundle_dir, meta): ...                              # optional
    @staticmethod
    def simulate_resume_request(session_lines, sentinel): ...                         # optional, enables `asr selfcheck`
```

### The IR contract

`load_trace` must return the dict shape in [design.md §1](design.md#trajectory-ir-trace). The bundle builder and
replay planner read:

- top level: `trial`, `session_id`, `project_slug`, `cc_version` (agent version), `start_ts`, `flags`,
  `sources.session_jsonl` (path relative to the trial dir), `sources.session_sha256`, `steps`, `sidechains`;
- each step: `index`, `atif_step_id`, `source` (`agent`/`user`), `elapsed_s`, `line_range`, `events`;
- `tool_call` events: `call_id`, `tool`, `args`, `side_effect`, `cwd`, `duration_s`, `is_error`, and the background
  fields if the agent has background jobs; `observation` events: `call_id`, `text`, `is_error`.

**Map the agent's tools onto the replayer's vocabulary** in `load_trace`. The in-sandbox replayer
(`replay/replay_inside.py`) understands:

| Replayer tool | `args` | Notes |
|---|---|---|
| `Write` | `file_path`, `content` | exact write |
| `Edit` / `MultiEdit` | `file_path`, `old_string`, `new_string`, `replace_all` | exact replace |
| `Bash` | `command`, `timeout` (ms), `run_in_background` | output compared with the original observation |
| anything with `side_effect` `read_only` / `web` / `task_mgmt` | – | skipped / flagged |

For example, a Codex `apply_patch` call can be translated into `Edit`/`Write` actions, and its `shell` call into
`Bash`. The replayer compares Bash output using Claude Code's conventions (e.g. an `Exit code N` prefix on failure,
stdout followed by stderr). If your agent formats tool results differently, normalize them in `load_trace` or extend
`compare()` in the replayer. Keep the replayer standard-library only and free of package imports, because it is
uploaded as a single file.

### The cut line

`cut_line` returns how many leading lines of the native session form the prefix. The prefix must end with the last
tool result before step *k*, and must not contain bookkeeping lines that would make the agent think a new turn
started. Truncation is byte-exact: `session_trunc.jsonl` is `"\n".join(lines[:cut]) + "\n"`. If the agent's session
is not line-oriented (e.g. one JSON document), store a byte-exact truncated copy and document how it was cut.

## 3. Sandbox side: a Harbor agent class

```python
from harbor.agents.installed.<agent> import <HarborAgent>, <HarborAgentOptions>
from .base import SnapshotResumeAgentMixin, bounded_remote_command

class CodexSnapshotResumeOptions(<HarborAgentOptions>):
    resume_bundle: str | None = None
    upstream_fake: bool = False
    upstream_url: str | None = None

class CodexSnapshotResume(SnapshotResumeAgentMixin, <HarborAgent>):
    options_model = CodexSnapshotResumeOptions
    adapter = CodexAdapter()

    async def install(self, environment):
        # 1. verify the snapshot's agent version == self._version (never install: it would change the env)
        # 2. self.adapter.sanity_check_bundle(...)
        # 3. await self._start_proxies(environment, bundle, meta)

    async def run(self, instruction, environment, context):
        # 1. env: base URL -> http://127.0.0.1:4031/r/<run_id> (add /v1 if the agent expects it), DUMMY key,
        #    drop every real credential from the env Harbor resolved
        # 2. upload session_trunc.jsonl + aux files to where the agent looks for them
        # 3. await self._verify_seeded(environment, {remote: local, ...}, env)
        # 4. run the resume command with meta["sentinel"] as the prompt, wrapped in
        #    bounded_remote_command(cmd, meta["agent_budget_s"])
```

### Wire API

Existing specs default to Anthropic Messages. Native specs select `responses` or `chat`
in `proxy_spec.json.wire`. The handler requires the matching endpoint and an exact
sentinel user turn in the first request. A hint replaces that turn at its original
position. Every other request field participates in a full captured-body comparison.
The passthrough uses Bearer authentication on the two OpenAI endpoints.

The legacy chained-hint and rendered-attachment logic applies to Claude Code.
For native `asr equiv`, pass the bundle's exact `session_trunc.jsonl` and
`cut_line`; the prefix hash binds the reference to those bytes. The offline CLI
probes exercise the real renderers. They do not rely on a simulated request template.

## 4. Prove it

1. Unit tests: a small **synthetic** session fixture for the agent (hand-written, not copied from a real run) under
   `examples/fixtures/`, plus tests for bundle → simulate → rewrite → equiv, like `tests/test_bundle_e2e.py`.
2. `asr launch --fake --no-verifier` on a real snapshot, then `asr equiv` on the captured
   `first_request_rewritten.json`. Only declared extras are allowed.
3. Compare the raw first request with a request from a fresh run of the same version to find any other resume-time
   injections. Add them to `resume_artifacts` (and to the rewrite if they are structural).
4. Add the adapter to `ADAPTERS` and a row to the README.

The Codex adapter reads the selected turn's sandbox policy. A recorded unrestricted guest
with approval policy `never` and a disabled inner permission profile resumes with that
same policy inside the task sandbox. The proxy verifies the exact pinned CLI context for
that mode. A bundle whose seed records different permissions must be rebuilt.

## Known limitations

- Pi and Codex strict replay refuses workspace actions for tasks that specify a non-root agent user. The source replayer runs as root. A cut with no workspace actions can still be used.
- Codex refuses a prefix containing a potentially writing call under a restricted or unrecorded sandbox policy, or with an unverified permission override. It also refuses a response containing multiple calls that may write. Read-only parallel calls are supported.
- Pi refuses prefixes containing an unconfirmed write or edit, a cancelled manual command, or a Bash command that references a Pi variable whose value cannot be recovered. A uniquely matched argument-validation failure with a missing required field has no replay action.
- Native bundles can still be built to inspect their prefix and compare a captured request. A reason in `plan.json` under `replay_refusals` blocks workspace execution under both replay policies. Rebuild older native plans from the source trial before preparing a snapshot.
