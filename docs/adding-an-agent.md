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

The resume proxy rewrites Anthropic `/v1/messages` bodies (`messages[].content[]` blocks) and forwards everything
else unchanged. The passthrough proxy works for any HTTP API, because it only swaps the auth header and streams the
body.

An agent that uses another API, such as OpenAI Responses or Chat Completions, needs one of these:

- an upstream that serves Anthropic `/v1/messages`, with the agent configured to use it; or
- a `rewrite()` variant for that API's message shape. Add it under `proxy/`, select it from `proxy_spec.json`, and
  keep the rule the same: remove only the exact resume artifacts and the sentinel turn, and change nothing else.

## 4. Prove it

1. Unit tests: a small **synthetic** session fixture for the agent (hand-written, not copied from a real run) under
   `examples/fixtures/`, plus tests for bundle → simulate → rewrite → equiv, like `tests/test_bundle_e2e.py`.
2. `asr launch --fake --no-verifier` on a real snapshot, then `asr equiv` on the captured
   `first_request_rewritten.json`. Only declared extras are allowed.
3. Compare the raw first request with a request from a fresh run of the same version to find any other resume-time
   injections. Add them to `resume_artifacts` (and to the rewrite if they are structural).
4. Add the adapter to `ADAPTERS` and a row to the README.
