"""Offline test of the key-injecting passthrough proxy against a local fake upstream."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent_snapshot_resume.proxy import runtime

spec = importlib.util.spec_from_file_location("passthrough_proxy", runtime.HERE / "passthrough_proxy.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


class Passthrough(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seen = []

        async def upstream(request):
            self.seen.append((dict(request.headers), await request.read()))
            return web.json_response({"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                                      "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                      "usage": {"input_tokens": 1, "output_tokens": 1}})
        up = web.Application()
        up.router.add_post("/v1/messages", upstream)
        self.up = TestServer(up)
        await self.up.start_server()
        self.tmp = tempfile.TemporaryDirectory()
        args = pp.build_parser().parse_args(["--port", "1", "--key-file", "/dev/null",
                                             "--upstream", str(self.up.make_url("")).rstrip("/"),
                                             "--log-dir", self.tmp.name])
        app = web.Application()
        app["key"] = "real-key-in-memory"
        app["strip_is_error"] = app["strip_thinking"] = False
        app["drop_fields"] = []
        app["retry_403_marker"] = b""
        app["upstream"], app["proxy"], app["port"] = args.upstream, None, 1
        app["cafile"] = None
        app["log_dir"], app["keep_thinking_text"] = self.tmp.name, True
        app.on_startup.append(pp.on_startup)
        app.on_cleanup.append(pp.on_cleanup)
        app.router.add_route("*", "/{tail:.*}", pp.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.up.close()
        self.tmp.cleanup()

    async def test_key_injected_body_forwarded_unchanged_and_not_logged(self):
        body = b'{"model":"m","messages":[{"role":"user","content":"hi"}],"metadata":{"user_id":"x"}}'
        r = await self.client.post("/v1/messages", data=body,
                                   headers={"x-api-key": "dummy-resume", "content-type": "application/json"})
        self.assertEqual(r.status, 200, await r.text())
        headers, got = self.seen[0]
        self.assertEqual(got, body)
        self.assertEqual(headers.get("x-api-key"), "real-key-in-memory")
        self.assertNotIn("dummy-resume", json.dumps(headers))
        for p in Path(self.tmp.name).rglob("*"):
            if p.is_file():
                self.assertNotIn(b"real-key-in-memory", p.read_bytes())


class KeyFile(unittest.TestCase):
    def test_group_readable_key_file_refused(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("k")
        try:
            os.chmod(f.name, 0o644)
            with self.assertRaises(SystemExit):
                pp.read_key(f.name)
            os.chmod(f.name, 0o600)
            self.assertEqual(pp.read_key(f.name), "k")
        finally:
            os.unlink(f.name)
