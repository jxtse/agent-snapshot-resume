"""Make a genuine Pi session, record original provider bodies and probe native resumes.

Run with PYTHONPATH=src. All state (including the agent's home) is under --out.
"""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_snapshot_resume.agents.pi import pi_args, pi_models
from agent_snapshot_resume.proxy.native_wire import body_reference


TOOLS = [
    ("read", {"path": "input.txt"}),
    ("write", {"path": "output.txt", "content": "one\n"}),
    ("edit", {"path": "output.txt", "edits": [{"oldText": "one", "newText": "two"}]}),
    ("bash", {"command": "cat output.txt"}),
]


@contextmanager
def upstream(script, capture_dir=None):
    bodies = []
    hashes = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            hashes.append(hashlib.sha256(raw).hexdigest())
            body = json.loads(raw)
            index = len(bodies)
            if capture_dir is not None:
                (capture_dir / f"request-{index}.json").write_bytes(raw)
            bodies.append(body)
            if index < len(script):
                name, args = script[index]
                delta = {"tool_calls": [{"index": 0, "id": f"call_{index}", "type": "function",
                                         "function": {"name": name, "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            else:
                delta, finish = {"content": "Done"}, "stop"
            events = [{"id": f"r{index}", "object": "chat.completion.chunk", "model": "fixture",
                       "created": 1, "choices": [{"index": 0, "delta": d, "finish_reason": f}]}
                      for d, f in [({"role": "assistant", **delta}, None), ({}, finish)]]
            data = ("".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", bodies, hashes
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_pi(root, cwd, session, endpoint, prompt):
    home = root / "home"
    cfg = home / ".pi/agent"
    cfg.mkdir(parents=True, exist_ok=True)
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    (cfg / "models.json").write_text(json.dumps(pi_models(endpoint, "fixture", "local")))
    env = dict(HOME=str(home), PATH=os.environ["PATH"], TMPDIR=str(tmp), PI_OFFLINE="1",
               PI_CODING_AGENT_DIR=str(cfg), HTTP_PROXY="http://127.0.0.1:1",
               HTTPS_PROXY="http://127.0.0.1:1", ALL_PROXY="http://127.0.0.1:1", NO_PROXY="127.0.0.1",
               DISABLE_TELEMETRY="1", SHELL="/bin/bash")
    return subprocess.run(["pi", *pi_args(session, "fixture", "local", prompt)], env=env, cwd=cwd,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90).returncode


def main():
    from agent_snapshot_resume.agents.pi import PiAdapter
    from agent_snapshot_resume.bundle import build
    from agent_snapshot_resume.parity import check_files
    from agent_snapshot_resume.proxy import resume_proxy
    from aiohttp import ClientSession, web

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    opts = ap.parse_args()
    root = opts.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    trial, cwd = root / "trial", root / "work"
    session = trial / "agent/pi/sessions/original.jsonl"
    session.parent.mkdir(parents=True)
    cwd.mkdir()
    (cwd / "input.txt").write_text("fixture\n")
    captures = trial / "agent/model-proxy"
    captures.mkdir()
    with upstream(TOOLS, captures) as (url, originals, hashes):
        rc = run_pi(root, cwd, session, url, "Exercise the fixture tools.")
    if rc or len(originals) != 5 or (cwd / "output.txt").read_text() != "two\n":
        raise RuntimeError(f"fixture generation failed (rc={rc}, requests={len(originals)})")
    records = [json.loads(line) for line in session.read_text().splitlines()]
    assistants = [r for r in records if r.get("message", {}).get("role") == "assistant"]
    results = [r["message"] for r in records if r.get("message", {}).get("role") == "toolResult"]
    assert len(assistants) == 5 and [r["toolName"] for r in results] == [name for name, _ in TOOLS]
    assert all(not r["isError"] for r in results)
    with (captures / "native-reference.jsonl").open("w") as f:
        for assistant, original, checksum in zip(assistants, originals, hashes, strict=True):
            ref = dict(body_reference("chat", original), raw_sha256=checksum)
            print(json.dumps({"entry_id": assistant["id"], "session_id": records[0]["id"], "content": ref}), file=f)
    version = subprocess.check_output(["pi", "--version"], text=True, env={"PATH": os.environ["PATH"],
                                      "HOME": str(root/"home"), "PI_OFFLINE": "1"}).strip()
    (trial / "agent/pi/version.txt").write_text(version + "\n")
    for index in (1, 4):
        ir = PiAdapter().load_trace(trial, 3600)
        step = next(s for s in ir["steps"] if s.get("message_id") == assistants[index]["id"])
        b, meta = build(adapter=PiAdapter(), trial_dir=trial, atif_step=step["atif_step_id"],
                        out=root / "bundles", agent_timeout=3600)
        seeded = b / "resume.jsonl"
        shutil.copyfile(b / "session_trunc.jsonl", seeded)
        async def resumed(endpoint):
            run_id = meta["run_id"]
            specs = b / "specs"
            specs.mkdir()
            spec = json.loads((b / "proxy_spec.json").read_text())
            spec.update(upstream=endpoint.removesuffix("/v1"), first_request_parity=meta["orig_request_at_k"])
            (specs / f"{run_id}.json").write_text(json.dumps(spec))
            app = web.Application()
            async with ClientSession() as client:
                app.update(spec_dir=specs, log_dir=b / "logs", specs={}, counts={}, compact_seen={}, session=client)
                app.router.add_route("*", "/r/{run_id}/{tail:.*}", resume_proxy.handle)
                runner = web.AppRunner(app)
                await runner.setup()
                try:
                    await web.TCPSite(runner, "127.0.0.1", 0).start()
                    proxy = f"http://127.0.0.1:{runner.addresses[0][1]}/r/{run_id}/v1"
                    return await asyncio.to_thread(run_pi, root, cwd, seeded, proxy, meta["sentinel"])
                finally:
                    await runner.cleanup()

        with upstream([]) as (url, bodies, forwarded):
            rc = asyncio.run(resumed(url))
        logs = b / "logs" / meta["run_id"]
        path = logs / "first_request_rewritten.json"
        report = check_files(b, path) if path.exists() else json.loads((logs/"first_request_parity.json").read_text())
        same_bytes = forwarded == [hashes[index]]
        print(json.dumps(dict(step=meta["resume_atif_step"], line=meta["cut_line"], cli_rc=rc,
                              emitted=(logs/"first_request_raw.json").exists(), ok=report["ok"],
                              forwarded_bytes_match=same_bytes, problems=report.get("content_problems"))))
        if rc or not report["ok"] or not same_bytes:
            raise RuntimeError("Pi resume failed")



if __name__ == "__main__":
    main()
