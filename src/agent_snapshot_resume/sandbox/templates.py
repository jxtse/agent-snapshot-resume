"""Register a snapshot-capable template from a task's container image (FC sandbox "micro" build mode).

The task image is read from task.toml `[environment] docker_image` unless --image is given. Private registries use
REGISTRY_USERNAME / REGISTRY_PASSWORD from the environment. The template name is
"<owner tag>-<task>-<timestamp>" (digit-leading names get a "t" prefix).
"""
from __future__ import annotations

import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from ..config import owner_tag
from .fc_env import template_alias

# FC sandbox build options: second-generation micro VM (supports snapshots) and the disk size in MiB. FC only
# accepts these disk sizes; a task asking for more memory/disk must still be built with the largest legal disk.
FC_BUILD_HEADERS = {"X-E2B-Template-Build-Mode": "micro", "X-E2B-Template-Alpha-Disk-Size": "{disk_mib}"}
FC_DISK_MIB = (512, 10240, 15360)


def task_image(task_dir: Path) -> str:
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    image = (cfg.get("environment") or {}).get("docker_image")
    if not image:
        raise ValueError(f"{task_dir}/task.toml has no [environment] docker_image; pass --image")
    return image


def fc_disk_mib(storage_mb: int | None) -> int:
    """The FC disk size for a task's storage request (smallest legal bucket that covers it, else the largest)."""
    want = int(storage_mb or FC_DISK_MIB[-1])
    for bucket in FC_DISK_MIB:
        if bucket >= want:
            return bucket
    return FC_DISK_MIB[-1]


def task_resources(task_dir: Path) -> dict[str, int]:
    """The cpu/memory/disk the task's own `[environment]` asks for (the trial ran with these).

    A replay sandbox built with fewer cpus or less memory does not reproduce the trial: torch scales its thread pool
    to the visible cpu count (4 -> 2 changes numerics) and a solver that fit in the trial's memory is OOM-killed."""
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    env = cfg.get("environment") or {}
    return {"cpus": int(env.get("cpus") or 2), "memory_mb": int(env.get("memory_mb") or 4096),
            "storage_mb": int(env.get("storage_mb") or FC_DISK_MIB[-1]),
            "disk_mib": fc_disk_mib(env.get("storage_mb"))}


def template_name(task: str, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%dt%H%M%S", time.localtime(now)).lower()
    name = f"{owner_tag()}-{re.sub(r'[^a-z0-9-]', '-', task.lower())}-{stamp}"
    return template_alias(name)


def build_template(*, task_dir: Path, state_dir: Path, image: str | None = None, cpu: int | None = None,
                   memory_mb: int | None = None, disk_mib: int | None = None, poll_s: int = 10,
                   max_wait_s: int = 1200) -> dict[str, Any]:
    from e2b import Template

    task = task_dir.name
    image = image or task_image(task_dir)
    # Resources default to the task's own [environment] block: the trial config and task.toml are the ground truth.
    res = task_resources(task_dir)
    cpu = res["cpus"] if cpu is None else cpu
    memory_mb = res["memory_mb"] if memory_mb is None else memory_mb
    disk_mib = res["disk_mib"] if disk_mib is None else fc_disk_mib(disk_mib)
    name = template_name(task)
    state_dir.mkdir(parents=True, exist_ok=True)
    out = state_dir / f"template-{task}.json"
    rec: dict[str, Any] = {"task": task, "image": image, "name": name, "cpu": cpu, "memory_mb": memory_mb,
                           "disk_mib": disk_mib, "mode": "micro", "started": time.time()}
    user, password = os.environ.get("REGISTRY_USERNAME"), os.environ.get("REGISTRY_PASSWORD")
    base = Template().from_image(image, username=user, password=password) if user and password \
        else Template().from_image(image)
    headers = {k: v.format(disk_mib=disk_mib) for k, v in FC_BUILD_HEADERS.items()}
    build = Template.build_in_background(base, name=name, cpu_count=cpu, memory_mb=memory_mb,
                                         api_headers=headers, request_timeout=300)
    rec.update(template_id=build.template_id, build_id=build.build_id)
    out.write_text(json.dumps(rec, indent=1))
    logs, off, status, st = [], 0, "unknown", None
    for _ in range(max(1, max_wait_s // poll_s)):
        st = Template.get_build_status(build, logs_offset=off)
        off += len(st.log_entries)
        logs += [e.message[:300] for e in st.log_entries]
        status = str(st.status).lower()
        if "ready" in status or "error" in status or "fail" in status:
            break
        time.sleep(poll_s)
    rec.update(status=status, seconds=round(time.time() - rec["started"], 1), logs=logs[-30:],
               reason=(st.reason.message if st is not None and st.reason else None))
    out.write_text(json.dumps(rec, indent=1))
    return rec
