"""Key-injecting Anthropic /v1/messages passthrough proxy (runs inside the sandbox, aiohttp only).

The agent talks to this proxy with a dummy key; the proxy:
  1. injects the real API key, read once from --key-file (a pipe from the start script: the key is never on disk);
  2. forwards the request body byte-for-byte to --upstream (unless an optional compatibility flag is given);
  3. logs one JSONL record per request (status, latency, ttfb, session headers, sizes, thinking block stats);
  4. retries connection errors before any byte was sent, and optionally a transient 403 whose body contains
     --retry-403-marker (some gateways answer 403 to a dispatch glitch; Claude Code treats 403 as fatal).

    python passthrough_proxy.py --port 4013 --key-file <(printf '%s' "$KEY") --upstream https://api.example.com \
        --log-dir /logs/agent/model-proxy
"""
import argparse, asyncio, json, logging, os, ssl, time

import aiohttp
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding", "keep-alive"}
DISPATCH_RETRIES = 6
CONNECT_RETRIES = 4  # 5+10+15+20 s: covers TLS/connect blips
log = logging.getLogger("asr-passthrough")
STATS = {"requests": 0, "errors": 0, "status": {}}
AUTH = {"x-api-key", "authorization"}


class ThinkTap:
    """Passively copies thinking blocks out of a /v1/messages response (SSE or JSON) while it streams
    to the client unchanged. One JSONL record per response; join to agent logs via msg_id."""

    def __init__(self, sse: bool):
        self.sse, self.buf, self.raw = sse, b"", bytearray()
        self.blocks: dict[int, dict] = {}
        self.msg: dict = {}

    def feed(self, chunk: bytes) -> None:
        if not self.sse:
            if len(self.raw) < 64 * 1024 * 1024:
                self.raw += chunk
            return
        self.buf = (self.buf + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.buf:
            ev, self.buf = self.buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data:"):
                    try:
                        self._event(json.loads(line[5:]))
                    except ValueError:
                        pass

    def _event(self, d: dict) -> None:
        t = d.get("type")
        if t == "message_start":
            m = d.get("message") or {}
            self.msg.update(msg_id=m.get("id"), model=m.get("model"), usage_start=m.get("usage"))
        elif t == "content_block_start":
            cb = d.get("content_block") or {}
            if cb.get("type") in ("thinking", "redacted_thinking"):
                self.blocks[d.get("index")] = {"index": d.get("index"), "type": cb["type"],
                                               "thinking": cb.get("thinking") or "",
                                               "signature": cb.get("signature") or "", "data": cb.get("data")}
        elif t == "content_block_delta":
            b, dl = self.blocks.get(d.get("index")), d.get("delta") or {}
            if b is not None:
                if dl.get("type") == "thinking_delta":
                    b["thinking"] += dl.get("thinking") or ""
                elif dl.get("type") == "signature_delta":
                    b["signature"] += dl.get("signature") or ""
        elif t == "message_delta":
            self.msg["stop_reason"] = (d.get("delta") or {}).get("stop_reason")
            self.msg["usage"] = d.get("usage")

    def record(self) -> dict:
        if not self.sse and self.raw:
            try:
                m = json.loads(bytes(self.raw))
                self.msg.update(msg_id=m.get("id"), model=m.get("model"), stop_reason=m.get("stop_reason"),
                                usage=m.get("usage"))
                for i, b in enumerate(m.get("content") or []):
                    if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
                        self.blocks[i] = {"index": i, "type": b["type"], "thinking": b.get("thinking") or "",
                                          "signature": b.get("signature") or "", "data": b.get("data")}
            except ValueError:
                pass
        blocks = [dict(b, signature_len=len(b.pop("signature") or "")) for _, b in sorted(self.blocks.items())]
        return dict(self.msg, n_thinking_blocks=len(blocks),
                    thinking_chars=sum(len(b["thinking"]) for b in blocks), thinking=blocks)



def jlog(app, name, rec):
    with open(os.path.join(app["log_dir"], time.strftime(f"{name}-%Y%m%d.jsonl")), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    STATS["requests"] += 1
    body = await request.read()
    meta = {}
    dropped = []
    # Optional compatibility rewrites for upstreams that reject some Anthropic fields. All are off by default, and
    # with all of them off the request body is forwarded byte-for-byte.
    if (app["drop_fields"] or app["strip_is_error"] or app["strip_thinking"]) and request.method == "POST" \
            and request.path.startswith("/v1/messages") and body:
        try:
            _d = json.loads(body)
            dropped = [k for k in app["drop_fields"] if k in _d]
            for k in dropped:
                _d.pop(k)
            n_err = n_think = 0
            for m in _d.get("messages") or []:
                cont = m.get("content")
                if not isinstance(cont, list):
                    continue
                keep = []
                for b in cont:
                    if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") and app["strip_thinking"]:
                        n_think += 1
                        continue
                    if isinstance(b, dict) and b.get("type") == "tool_result" and "is_error" in b and app["strip_is_error"]:
                        if b.pop("is_error"):
                            n_err += 1
                    keep.append(b)
                m["content"] = keep
            if n_err: dropped.append(f"is_error_true:{n_err}")
            if n_think: dropped.append(f"thinking_blocks:{n_think}")
            body = json.dumps(_d, ensure_ascii=False).encode()
        except ValueError:
            pass
    if request.method == "POST" and request.path.startswith("/v1/messages") and body:
        try:
            d = json.loads(body)
            msgs = d.get("messages") or []
            meta = {"model": d.get("model"), "n_messages": len(msgs), "max_tokens": d.get("max_tokens"),
                    "thinking": d.get("thinking"), "n_tools": len(d.get("tools") or []), "req_bytes": len(body),
                    "n_replayed_thinking": sum(1 for m in msgs if isinstance(m.get("content"), list)
                                               for b in m["content"] if isinstance(b, dict) and b.get("type") == "thinking")}
        except (ValueError, AttributeError):
            pass
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP | AUTH}
    headers["x-api-key"] = app["key"]
    sess_hdr = {k: v for k, v in request.headers.items() if k.lower().startswith("x-claude-code") or k.lower() == "x-app"}
    t0 = time.time()
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "port": app["port"], "path": request.path, **meta, "session": sess_hdr, "dropped": dropped}
    retries = 0
    connect_retries = 0
    try:
        for attempt in range(DISPATCH_RETRIES + 1):
            up = None
            for c_try in range(CONNECT_RETRIES + 1):
                try:
                    up = await app["session"].request(request.method, app["upstream"] + request.path_qs, data=body,
                                                      headers=headers, proxy=app["proxy"])
                    break
                except aiohttp.ClientConnectorError as e:  # includes ClientProxyConnectionError
                    # Failed before any request bytes reached the upstream (TLS/connect blip): safe to retry.
                    if c_try == CONNECT_RETRIES:
                        raise
                    connect_retries += 1
                    STATS["connect_retried"] = STATS.get("connect_retried", 0) + 1
                    log.warning("connect error %r, retry %d after %ds", e, c_try + 1, 5 * (c_try + 1))
                    await asyncio.sleep(5 * (c_try + 1))
            if up.status == 403 and attempt < DISPATCH_RETRIES:
                txt = await up.read()
                up.release()
                if app["retry_403_marker"] and app["retry_403_marker"] in txt:
                    retries += 1
                    log.warning("transient 403, retry %d", retries)
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                rec.update(status=403, latency_s=round(time.time() - t0, 1), err=txt[:300].decode(errors="replace"))
                jlog(app, "requests", rec)
                return web.Response(status=403, body=txt, headers={k: v for k, v in up.headers.items()
                                                                   if k.lower() not in HOP | {"content-encoding"}})
            break
        async with up:
            resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items()
                                                                  if k.lower() not in HOP | {"content-encoding"}})
            enc = (up.headers.get("content-encoding") or "identity").lower()
            tap = ThinkTap("event-stream" in (up.headers.get("content-type") or "")) \
                if request.path == "/v1/messages" and up.status == 200 and enc == "identity" else None
            errbuf = b"" if up.status != 200 else None
            await resp.prepare(request)
            ttfb = None
            try:
                async for chunk in up.content.iter_any():
                    if ttfb is None:
                        ttfb = round(time.time() - t0, 1)
                    if tap is not None:
                        try:
                            tap.feed(chunk)
                        except Exception:  # noqa: BLE001
                            tap = None
                    if errbuf is not None and len(errbuf) < 600:
                        errbuf += chunk
                    await resp.write(chunk)
                await resp.write_eof()
            finally:
                rec.update(status=up.status, latency_s=round(time.time() - t0, 1), ttfb_s=ttfb, dispatch_retries=retries,
                           connect_retries=connect_retries)
                if errbuf:
                    rec["err"] = errbuf[:600].decode(errors="replace")
                if tap is not None:
                    try:
                        r = tap.record()
                        rec.update(upstream_model=r.get("model"), stop_reason=r.get("stop_reason"), usage=r.get("usage"),
                                   n_thinking_blocks=r.get("n_thinking_blocks"), thinking_chars=r.get("thinking_chars"),
                                   complete=bool(r.get("stop_reason")))
                        if app["keep_thinking_text"]:
                            jlog(app, "thinking", {"ts": rec["ts"], "session": sess_hdr, "thinking": r.get("thinking")})
                    except Exception as e:  # noqa: BLE001
                        rec["tap_error"] = repr(e)[:200]
                STATS["status"][str(up.status)] = STATS["status"].get(str(up.status), 0) + 1
                jlog(app, "requests", rec)
            return resp
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        rec.update(status="upstream_error", latency_s=round(time.time() - t0, 1), err=repr(e)[:300],
                   connect_retries=connect_retries)
        jlog(app, "requests", rec)
        log.warning("upstream error after %.1fs: %r", time.time() - t0, e)
        raise


async def health(request):
    return web.json_response({**STATS, "upstream": request.app["upstream"], "via_proxy": bool(request.app["proxy"])})


async def on_startup(app):
    ctx = ssl.create_default_context(cafile=app["cafile"]) if app["cafile"] else ssl.create_default_context()
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=3700),
                                           auto_decompress=False, connector=aiohttp.TCPConnector(ssl=ctx, limit=128))


async def on_cleanup(app):
    await app["session"].close()


def read_key(path):
    """Read the key once. Regular files must not be group/other readable; pipes (/dev/fd/N) are accepted."""
    st = os.stat(path)
    if st.st_mode & 0o077:
        raise SystemExit(f"{path} must not be readable by group/other")
    with open(path) as f:
        key = f.read().strip()
    if not key:
        raise SystemExit("empty key")
    return key


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--upstream", required=True, help="base URL of an Anthropic-messages-compatible endpoint")
    ap.add_argument("--proxy", default="", help="outbound HTTP proxy ('' = direct)")
    ap.add_argument("--cafile", default=None, help="CA bundle (default: system store)")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--no-thinking-text", action="store_true", help="log thinking sizes only, not text")
    ap.add_argument("--drop-fields", default="", help="comma-separated top-level body keys to strip")
    ap.add_argument("--strip-is-error", action="store_true")
    ap.add_argument("--strip-thinking", action="store_true")
    ap.add_argument("--retry-403-marker", default="", help="retry a 403 whose body contains this text")
    return ap


def main():
    a = build_parser().parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["key"] = read_key(a.key_file)
    app["strip_is_error"], app["strip_thinking"] = a.strip_is_error, a.strip_thinking
    app["drop_fields"] = [x for x in a.drop_fields.split(",") if x]
    app["retry_403_marker"] = a.retry_403_marker.encode() if a.retry_403_marker else b""
    app["upstream"], app["proxy"], app["port"] = a.upstream.rstrip("/"), a.proxy or None, a.port
    app["cafile"] = a.cafile
    app["log_dir"], app["keep_thinking_text"] = a.log_dir, not a.no_thinking_text
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/_health", health)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host=a.host, port=a.port, access_log=None, reuse_port=True, shutdown_timeout=120)


if __name__ == "__main__":
    main()
