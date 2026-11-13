"""Shell snippets that install and start the in-sandbox proxies.

Runtime: python-build-standalone CPython 3.12.12 (release 20251014, sha256-checked) plus aiohttp and its
dependencies installed from PyPI with --require-hashes (proxy-requirements.txt). The interpreter is copied to
`asr-model-proxy` under /opt/asr-proxy/rt, so an agent's `pkill python` / `killall python3` does not hit it.
The runtime is installed while the snapshot source is prepared (no key involved), so clones already have it.

Credential handling (`MEMORY_PROXY_START`): the key arrives as the env var ASR_PROXY_KEY of ONE exec call (it is
redacted from Harbor's logs because the name contains KEY). The parent shell writes a restart-loop script that
contains no secret, starts it detached, and unsets the variable. The loop copies the key into a non-exported shell
variable, unsets the env var, and hands it to each proxy start through a pipe (`--key-file <(printf ...)`).
The key is never written to disk, never on a command line and never in the proxy process environment.
"""
from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROXY_DIR = "/opt/asr-proxy"
STATE_DIR = "/root/.asr-proxy"
RUNTIME = f"{PROXY_DIR}/rt/bin/asr-model-proxy"
PASSTHROUGH_PORT = 4013
RESUME_PORT = 4031
PASSTHROUGH_LOGDIR = "/logs/agent/model-proxy"
RESUME_LOGDIR = "/logs/agent/resume-proxy"
KEY_ENV_IN_SANDBOX = "ASR_PROXY_KEY"

# Files uploaded into the snapshot at preparation time (explicit list: never upload stray files).
PROXY_FILES = ("passthrough_proxy.py", "proxy-requirements.txt")

PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20251014/"
           "cpython-3.12.12%2B20251014-x86_64-unknown-linux-gnu-install_only.tar.gz")
PBS_SHA256 = "1ab2b6594d1c3d76cbebea09d6bc3e6ba68d8eb3b6322080375c4cc3dd188f34"  # GitHub release asset digest


def fetch(url: str, dst: str, sha256: str) -> str:
    """Shell: download url to dst (curl, retried) and require sha256 == expected."""
    return (f"(for i in 1 2 3 4 5; do curl -fsSL --retry 3 --connect-timeout 30 -o {dst} {url} && break; "
            f"echo 'download attempt '$i' failed: {url}' >&2; sleep 10; done; "
            f"echo '{sha256}  {dst}' | sha256sum -c - )")


APT = (
    "for i in 1 2 3; do "
    "  timeout -k 10 {t} sh -c 'apt-get update && apt-get install -y {pkgs}' && exit 0; "
    "  echo \"apt attempt $i failed rc=$?\" >&2; "
    "  pkill -9 -x apt-get; pkill -9 -x dpkg; rm -f /var/lib/dpkg/lock* /var/lib/apt/lists/lock "
    "/var/cache/apt/archives/lock; dpkg --configure -a || true; sleep 5; "
    "done; exit 1"
)


def proxy_file_bytes() -> dict[str, bytes]:
    return {name: (HERE / name).read_bytes() for name in PROXY_FILES}


def sha256sums(files: dict[str, bytes]) -> str:
    return "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items()))


# Run as root after the files in PROXY_FILES plus SHA256SUMS were uploaded to /tmp/.asr-proxy-src and the
# host's CA bundle to /tmp/.asr-cacert.pem. {fetch} = fetch(PBS_URL, "/tmp/.asr-pbs.tgz", PBS_SHA256).
PROXY_PLACE = r"""
set -eu
P=/opt/asr-proxy
rm -rf "$P"; mkdir -p "$P/src"
mv /tmp/.asr-proxy-src/* "$P/src/"; rm -rf /tmp/.asr-proxy-src
cd "$P/src" && sha256sum -c SHA256SUMS
{fetch}
mkdir -p "$P/rt" && tar xzf /tmp/.asr-pbs.tgz -C "$P/rt" --strip-components=1 && rm -f /tmp/.asr-pbs.tgz
"$P/rt/bin/python3.12" -c 'import sys; assert sys.version_info[:3] == (3, 12, 12), sys.version; print(sys.version)'
cp "$P/rt/bin/python3.12" "$P/rt/bin/asr-model-proxy"
PIP_CONFIG_FILE=/dev/null "$P/rt/bin/asr-model-proxy" -m pip install -q --disable-pip-version-check --no-deps --only-binary :all: --index-url https://pypi.org/simple \
  --require-hashes --target "$P/site" -r "$P/src/proxy-requirements.txt"
PYTHONPATH="$P/site:$P/src" "$P/rt/bin/asr-model-proxy" -c 'import aiohttp, passthrough_proxy; print("proxy runtime ok, aiohttp", aiohttp.__version__)'
"""

# The proxies need a CA bundle at a fixed path. Prefer the image's system bundle; fall back to the uploaded one.
CA_LINK = r"""
set -eu
P=/opt/asr-proxy
for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
  if [ -s "$f" ]; then ln -sf "$f" "$P/ca-bundle.crt"; break; fi
done
[ -s "$P/ca-bundle.crt" ] || cp /tmp/.asr-cacert.pem "$P/ca-bundle.crt"
rm -f /tmp/.asr-cacert.pem
ls -laL "$P/ca-bundle.crt"
"""

# Starts the passthrough proxy as root on 127.0.0.1:{port} under a restart loop. See the module docstring.
MEMORY_PROXY_START = r"""
set -eu
P=/opt/asr-proxy
umask 077
mkdir -p /root/.asr-proxy {logdir}
cat > /root/.asr-proxy/run-{port}.sh <<EOF
#!/bin/bash
set -u
unset KV
KV="\$ASR_PROXY_KEY"; unset ASR_PROXY_KEY
cd $P/src
while true; do
  PYTHONPATH=$P/site:$P/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $P/rt/bin/asr-model-proxy {script} \
    --host 127.0.0.1 --port {port} --key-file <(printf '%s' "\$KV") --upstream {upstream} --cafile $P/ca-bundle.crt {extra} \
    --log-dir {logdir} >> {logdir}/proxy-{port}.stderr 2>&1 < /dev/null
  echo "proxy exited rc=\$? at \$(date -u +%FT%TZ), restarting" >> {logdir}/proxy-{port}.restarts
  sleep 2
done
EOF
chmod 700 /root/.asr-proxy/run-{port}.sh
setsid nohup /root/.asr-proxy/run-{port}.sh > /dev/null 2>&1 < /dev/null &
unset ASR_PROXY_KEY
for i in $(seq 1 60); do
  if "$P/rt/bin/asr-model-proxy" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:{port}/_health',timeout=2)" 2>/dev/null; then
    echo "proxy {port} up; memory-only credentials"; exit 0; fi
  sleep 1
done
cat {logdir}/proxy-{port}.stderr >&2; echo "proxy {port} did not start" >&2; exit 1
"""

RESUME_PROXY_START = r"""
set -eu
P=/opt/asr-proxy
mkdir -p {logdir} $P/resume/specs /root/.asr-proxy
cat > /root/.asr-proxy/run-resume.sh <<EOF
#!/bin/bash
cd $P/resume
while true; do
  PYTHONPATH=$P/site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $P/rt/bin/asr-model-proxy resume_proxy.py \
    --host 127.0.0.1 --port {port} --spec-dir $P/resume/specs --log-dir {logdir} >> {logdir}/resume-proxy.stderr 2>&1 < /dev/null
  echo "resume proxy exited rc=\$? at \$(date -u +%FT%TZ), restarting" >> {logdir}/resume-proxy.restarts
  sleep 2
done
EOF
chmod 700 /root/.asr-proxy/run-resume.sh
setsid nohup /root/.asr-proxy/run-resume.sh > /dev/null 2>&1 < /dev/null &
for i in $(seq 1 60); do
  if "$P/rt/bin/asr-model-proxy" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:{port}/_health',timeout=2)" 2>/dev/null; then
    echo "resume proxy up"; exit 0; fi
  sleep 1
done
cat {logdir}/resume-proxy.stderr >&2; echo "resume proxy did not start" >&2; exit 1
"""


# Extra passthrough-proxy flags an agent may request (e.g. the original route's `--drop-fields metadata
# --strip-is-error`). Whitelisted so a job config cannot inject shell or override host/port/key/upstream/log-dir.
PASSTHROUGH_FLAGS = {"--drop-fields": 1, "--strip-is-error": 0, "--strip-thinking": 0, "--retry-403-marker": 1,
                     "--no-thinking-text": 0}
_ARG_VALUE_RE = re.compile(r"[A-Za-z0-9_.,:-]+")


def validate_passthrough_args(args) -> list[str]:
    if isinstance(args, str):
        args = shlex.split(args)
    args, out, i = list(args), [], 0
    while i < len(args):
        flag = args[i]
        if flag not in PASSTHROUGH_FLAGS:
            raise ValueError(f"passthrough arg {flag!r} not allowed (allowed: {sorted(PASSTHROUGH_FLAGS)})")
        out.append(flag)
        if PASSTHROUGH_FLAGS[flag]:
            if i + 1 >= len(args) or not _ARG_VALUE_RE.fullmatch(args[i + 1]):
                raise ValueError(f"passthrough arg {flag} needs a plain value")
            out.append(args[i + 1])
            i += 1
        i += 1
    return out


def passthrough_start_command(upstream: str, port: int = PASSTHROUGH_PORT, logdir: str = PASSTHROUGH_LOGDIR,
                              extra: str = "") -> str:
    if not upstream.startswith(("http://", "https://")):
        raise ValueError("upstream must be an http(s) URL")
    return MEMORY_PROXY_START.format(script="passthrough_proxy.py", port=int(port), upstream=shlex.quote(upstream),
                                     extra=extra, logdir=logdir)


def resume_start_command(port: int = RESUME_PORT, logdir: str = RESUME_LOGDIR) -> str:
    return RESUME_PROXY_START.format(port=int(port), logdir=logdir)


def proxy_place_command() -> str:
    return PROXY_PLACE.format(fetch=fetch(PBS_URL, "/tmp/.asr-pbs.tgz", PBS_SHA256))
