"""Harbor's E2B environment, adjusted for Alibaba Cloud's Function Compute (FC) sandbox service (E2B-compatible API).

Use it in a Job config instead of Harbor's built-in `e2b` environment:

    environment:
      import_path: agent_snapshot_resume.sandbox.fc_env:SandboxEnvironment

What differs from the built-in one:

- Network rules are always sent as explicit egress rules. The service ignores the internet
  switch on a running sandbox (after `update_network(allow_internet_access=False)` a sandbox
  still reached PyPI), and that switch is how the built-in environment takes a sandbox offline,
  so a task that declares no network would run online. A deny-all rule plus an allow-list is
  honoured at creation and in an update. An update replaces the rules, so a public phase clears
  both lists: Harbor's own update for public is empty and would leave the agent phase's deny-all
  in place while the tests run. The switch itself stays on at creation.
- The model host is let through with Harbor's own `--allow-agent-host <host>`: Harbor merges it
  into the agent phase only, so the tests do not get it.
- Creation retries the errors that clear on their own (flow control, 412 "accelerated image not
  ready") up to 8 times; anything else keeps Harbor's two attempts.
- Template aliases that start with a digit are rejected by the service, so they get a "t" prefix
  (`template_alias()`; apply the same function when you register templates yourself).
- /tmp gets mode 1777 (Docker's default) in every sandbox; the service creates it as 0777.

VpcSandboxEnvironment also joins agent sandboxes to the VPC in FC_SANDBOX_VPC (a JSON string with
vpcId, vSwitchIds, securityGroupId), for a model service that has a private address only. The
service joins a sandbox to a VPC when its metadata carries `fc.sandbox.network.vpc`. Separate
verifier sandboxes (built from tests/) do not join. The first VPC sandbox of a template makes
the service create a function that queries the image registry, whose flow control rejects bursts, so
these creations are paced: FC_SANDBOX_CREATE_CONCURRENCY in flight (default 8), one start per
FC_SANDBOX_CREATE_INTERVAL seconds (default 0.25).
"""
import asyncio
import contextlib
import json
import os

from e2b import AsyncSandbox
from e2b.sandbox.network import ALL_TRAFFIC
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy
from tenacity import RetryCallState, retry, wait_random_exponential

VPC_METADATA_KEY = "fc.sandbox.network.vpc"
SANDBOX_TIMEOUT_SEC = 86_400  # Harbor's own lifetime for a sandbox nobody kills


def template_alias(alias: str) -> str:
    """The FC sandbox service rejects template aliases that start with a digit."""
    return ("t" + alias) if alias[:1].isdigit() else alias


def is_transient(error: BaseException) -> bool:
    message = str(error)
    return any(marker in message for marker in (
        "Throttling", "flow control", "accelerated image not ready"))


def stop_creating(state: RetryCallState) -> bool:
    error = state.outcome.exception() if state.outcome else None
    limit = 8 if error is not None and is_transient(error) else 2
    return state.attempt_number >= limit


class SandboxEnvironment(E2BEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template_name = template_alias(self._template_name)

    async def start(self, force_build: bool):
        await super().start(force_build)
        # Docker gives /tmp mode 1777; this service gives 0777, and Harbor runs the task healthcheck only in the
        # agent sandbox, not in a separate verifier sandbox. Restore the container default in every sandbox
        # (some verifiers check the sticky bit).
        await self.exec("chmod 1777 /tmp", user="root")

    def _sandbox_create_network_options(self, network_policy: NetworkPolicy | None = None):
        policy = network_policy or self.network_policy
        return {
            "allow_out": list(policy.allowed_hosts),
            "deny_out": [] if policy.network_mode == NetworkMode.PUBLIC else [ALL_TRAFFIC],
        }

    def _sandbox_network_update(self, network_policy: NetworkPolicy | None = None):
        return self._sandbox_create_network_options(network_policy)

    def _extra_metadata(self) -> dict[str, str]:
        return {}

    def _create_gate(self):
        return contextlib.nullcontext()

    @retry(stop=stop_creating, wait=wait_random_exponential(multiplier=2, max=60), reraise=True)
    async def _create_sandbox(self):
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            **self._extra_metadata(),
        }
        async with self._create_gate():
            self._sandbox = await AsyncSandbox.create(
                template=self._template_name,
                metadata=metadata,
                envs=self._startup_env(),
                timeout=SANDBOX_TIMEOUT_SEC,
                allow_internet_access=True,
                network=self._sandbox_create_network_options(),
            )


class CreateGate:
    """At most `slots` creations in flight, and consecutive starts at least `interval` apart."""

    def __init__(self, slots: int, interval: float):
        self.interval = interval
        self.slots = asyncio.Semaphore(slots)
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def __aenter__(self):
        await self.slots.acquire()
        async with self.lock:
            loop = asyncio.get_running_loop()
            delay = self.next_start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = loop.time() + self.interval

    async def __aexit__(self, *exc):
        self.slots.release()


_gates: dict[int, CreateGate] = {}


def vpc_metadata() -> str:
    raw = os.environ.get("FC_SANDBOX_VPC")
    if not raw:
        raise RuntimeError("FC_SANDBOX_VPC is required for VpcSandboxEnvironment; put it in .env")
    config = json.loads(raw)
    missing = {"vpcId", "vSwitchIds", "securityGroupId"} - config.keys()
    if missing:
        raise RuntimeError(f"FC_SANDBOX_VPC is missing {sorted(missing)}")
    return json.dumps(config, separators=(",", ":"))


class VpcSandboxEnvironment(SandboxEnvironment):
    @property
    def joins_vpc(self) -> bool:
        return self.environment_dir.name == "environment"  # a separate verifier is built from tests/

    def _extra_metadata(self) -> dict[str, str]:
        return {VPC_METADATA_KEY: vpc_metadata()} if self.joins_vpc else {}

    def _create_gate(self):
        if not self.joins_vpc:
            return contextlib.nullcontext()
        loop = asyncio.get_running_loop()  # one gate per event loop: every trial of a job shares it
        if id(loop) not in _gates:
            _gates[id(loop)] = CreateGate(
                int(os.environ.get("FC_SANDBOX_CREATE_CONCURRENCY", "8")),
                float(os.environ.get("FC_SANDBOX_CREATE_INTERVAL", "0.25")),
            )
        return _gates[id(loop)]
