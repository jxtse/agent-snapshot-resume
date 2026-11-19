"""Build real trial cuts and probe with the integrity-checked 0.154.0 macOS arm64 binary."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_snapshot_resume.agents.codex import CodexAdapter
from agent_snapshot_resume.bundle import build
from agent_snapshot_resume.parity import check_files
from agent_snapshot_resume.proxy.resume_proxy import rewrite


VERSION = "0.154.0"
REGISTRY = "https://registry.npmjs.org"


def verify_dist(archive, metadata, version):
    if metadata["name"] != "@openai/codex" or metadata["version"] != version:
        raise ValueError("registry metadata identifies a different release")
    expected = metadata["dist"]["integrity"]
    with archive.open("rb") as stream:
        hash_bytes = hashlib.file_digest(stream, "sha512").digest()
    actual = "sha512-" + base64.b64encode(hash_bytes).decode()
    if actual != expected:
        raise ValueError("registry SHA512 mismatch")


def pinned_cli(cache, *, cached_only=False):
    isolated = dict(PATH=os.environ["PATH"], HOME=str(cache/"home"), TMPDIR=str(cache/"tmp"),
                    npm_config_registry=REGISTRY, npm_config_cache=str(cache/"npm"),
                    npm_config_userconfig=os.devnull, npm_config_globalconfig=str(cache/"empty-npmrc"),
                    npm_config_update_notifier="false", npm_config_audit="false", npm_config_fund="false",
                    npm_config_fetch_retries="0")
    for sub in ("home", "tmp"):
        (cache/sub).mkdir(exist_ok=True)
    (cache/"empty-npmrc").touch()
    for version in (VERSION, VERSION+"-darwin-arm64"):
        meta_path = cache / f"codex-{version}-registry.json"
        archive = cache / f"openai-codex-{version}.tgz"
        if cached_only and not (archive.is_file() and meta_path.is_file()):
            raise FileNotFoundError(f"missing cached archive or metadata for {version}")
        if not archive.exists():
            result = subprocess.run(["npm", "pack", f"@openai/codex@{version}", "--ignore-scripts",
                                     "--pack-destination", str(cache)], cwd=cache, env=isolated,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
            if result.returncode:
                codes = [e for e in ("ENOTCONN", "ENOTFOUND", "ECONNRESET", "E403", "E404") if e.encode() in result.stderr]
                raise RuntimeError(f"npm pack {version} failed ({', '.join(codes) or result.returncode})")
        if not meta_path.exists():
            class RejectRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args):
                    raise ValueError("unexpected registry redirect")
            opener = urllib.request.build_opener(RejectRedirect)
            with opener.open(f"{REGISTRY}/@openai%2fcodex/{version}", timeout=30) as response:
                metadata = json.load(response)
            verify_dist(archive, metadata, version)
            meta_path.write_text(json.dumps(metadata))
        verify_dist(archive, json.loads(meta_path.read_text()), version)
    native = cache / "pinned-darwin-arm64"
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.isfile() and m.name.startswith("package/vendor/aarch64-apple-darwin/")
                   and m.name.endswith("/codex")]
        if len(members) != 1:
            raise ValueError("pinned archive has no unique native executable")
        with native.open("wb") as output:
            shutil.copyfileobj(tar.extractfile(members[0]), output)
        native.chmod(0o755)
    if subprocess.check_output([native, "--version"], env=isolated, text=True).strip() != "codex-cli " + VERSION:
        raise RuntimeError("pinned Codex binary version mismatch")
    return native


def first_trace_request(trace, session_id):
    path, = trace.glob("*/trace.jsonl")
    for line in path.read_text().splitlines():
        event = json.loads(line)["payload"]
        if event["type"] == "inference_started" and event["thread_id"] == session_id:
            payload = event["request_payload"]
            if payload["kind"]["type"] == "inference_request":
                return json.loads((path.parent / payload["path"]).read_bytes())
    raise ValueError("no traced first Responses request")


def gate(raw, bundle, local, filename):
    spec = json.loads((bundle / "proxy_spec.json").read_text())
    spec["resume_artifacts"].update(cwd=str(local / "work"), codex_home=str(local / "codex-home"))
    try:
        body, actions = rewrite(raw, spec, first=True)
    except ValueError as exc:
        return {"verdict": "mismatch", "ok": False, "reason": str(exc)}
    path = bundle / filename
    path.write_text(json.dumps(body))
    return dict(check_files(bundle, path), actions=actions)


def capture(cli, bundle, meta):
    CodexAdapter().sanity_check_bundle(bundle, meta)
    root = bundle.resolve() / "local"
    home = root / "codex-home"
    cwd = root / "work"
    cwd.mkdir(parents=True)
    seen = []

    class Capture(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/responses":
                self.send_error(404)
                return
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_error(400, "Probe recorded first request")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
    except PermissionError:
        server = None
    port = server.server_port if server else 1
    if server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
    try:
        env, _, args, _, config = CodexAdapter().resume_setup(meta, f"http://127.0.0.1:{port}", home=str(home))
        dest = home / Path(meta["session_relpath"]).relative_to("agent")
        dest.parent.mkdir(parents=True)
        shutil.copyfile(bundle / "session_trunc.jsonl", dest)
        for aux in (bundle / "aux").iterdir():
            shutil.copyfile(aux, home / aux.name)
        (home/"config.toml").write_text(config)
        env.update(HOME=str(root), CODEX_HOME=str(home), PATH=os.environ["PATH"], TMPDIR=str(root),
                   NO_PROXY="127.0.0.1", HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1",
                   ALL_PROXY="http://127.0.0.1:1", DISABLE_TELEMETRY="1")
        if server is None:
            env["CODEX_ROLLOUT_TRACE_ROOT"] = str(root/"trace")
        with subprocess.Popen([cli, *args[1:]], env=env, cwd=cwd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL) as process:
            stopped_after_trace = False
            deadline = time.monotonic() + 90
            while process.poll() is None:
                if server is None:
                    try:
                        first_trace_request(root / "trace", meta["session_id"])
                    except (OSError, ValueError):
                        pass
                    else:
                        stopped_after_trace = True
                        process.terminate()
                        break
                if time.monotonic() >= deadline:
                    process.kill()
                    break
                time.sleep(0.1)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        if server:
            server.shutdown()
            server.server_close()
            worker.join()
    if seen:
        raw = seen[0]
        source = "http"
    elif server is None:
        try:
            raw = first_trace_request(root / "trace", meta["session_id"])
        except (ValueError, OSError):
            return {"emitted": False, "ok": False, "cli_rc": process.returncode,
                    "reason": "loopback bind denied and CLI produced no complete request trace"}
        source = "inference_started"
    else:
        return {"emitted": False, "cli_rc": process.returncode, "reason": "no Responses request"}
    name = "first_request" if seen else "traced_request"
    (bundle / f"{name}_raw.json").write_text(json.dumps(raw))
    report = gate(raw, bundle, root, f"{name}_rewritten.json")
    return dict(report, emitted=bool(seen), capture_source=source, cli_rc=process.returncode,
                stopped_after_trace=stopped_after_trace)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--cached-only", action="store_true", help="refuse any package download")
    ap.add_argument("--prior-http", type=Path, help="also compare existing independent HTTP captures of the same cuts")
    ap.add_argument("--trial", required=True, action="append", help="trial directory name (repeatable)")
    ap.add_argument("--step", type=int, action="append", help="ATIF step (repeatable; defaults to 7 and 10)")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    try:
        cli, reason = pinned_cli(args.cache.resolve(), cached_only=args.cached_only), None
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        cli, reason = None, str(exc)
    adapter = CodexAdapter(args.cache / "deepseek-models.json")
    passed = True
    for trial in args.trial:
        for step in args.step or (7, 10):
            bundle, meta = build(adapter=adapter, trial_dir=args.trials/trial, atif_step=step,
                                 agent_timeout=7200, out=args.out)
            result = capture(cli, bundle, meta) if cli else {"emitted": False, "ok": None, "reason": reason}
            if args.prior_http:
                prior = args.prior_http.resolve() / bundle.name
                saved = json.loads((prior / "bundle.json").read_text())
                if any(meta[key] != saved[key] for key in ("session_id", "prefix_sha256", "sentinel", "cut_line")):
                    raise ValueError("prior capture belongs to a different cut")
                raw = json.loads((prior / "first_request_raw.json").read_bytes())
                result["prior_http"] = gate(raw, bundle, prior / "local", "prior_http_rewritten.json")
                if "all_items_equal" in result and "all_items_equal" in result["prior_http"]:
                    name = "first_request" if result["emitted"] else "traced_request"
                    fresh = json.loads((bundle / f"{name}_rewritten.json").read_bytes())
                    saved_http = json.loads((bundle / "prior_http_rewritten.json").read_bytes())
                    fields = (fresh.keys() | saved_http.keys()) - {"input"}
                    result["fresh_vs_prior_http"] = {
                        "ordered_input_equal": fresh["input"] == saved_http["input"],
                        "different_envelope_fields": sorted(k for k in fields if k not in fresh or
                                                            k not in saved_http or fresh[k] != saved_http[k]),
                    }
            print(json.dumps(dict(trial=trial, step=step, cut_line=meta["cut_line"], **result)))
            (bundle/"probe_result.json").write_text(json.dumps(result))
            passed = passed and bool(result.get("ok")) and (not args.prior_http or result["prior_http"]["ok"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
