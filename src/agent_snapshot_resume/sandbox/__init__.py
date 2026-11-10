"""E2B-compatible sandbox side (Alibaba Cloud FC sandbox by default).

fc_env.py            Harbor environment for the FC sandbox service (explicit egress, retries, VPC variant)
snapshot_env.py      SnapshotCloneEnvironment: agent sandbox cloned from a snapshot, with fail-closed gates
command_recovery.py  one-dispatch / same-PID-reconnect command execution used by the clone environment
templates.py         register a snapshot-capable template from a task image
source.py            prepare a source sandbox, replay steps < k, gate, fingerprint, snapshot, delete the source
manage.py            list / kill sandboxes carrying this tool's owner tag; delete snapshots
"""
