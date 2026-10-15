"""hfsandbox provider — the local auth-injecting proxy (HTTP, SSE, WebSocket)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments.providers.hfsandbox import (
    _AuthProxy,
    _build_provider_config,
    parse_volume_spec,
)

aiohttp = pytest.importorskip("aiohttp")
web = pytest.importorskip("aiohttp.web")

HEADERS = {"Authorization": "Bearer hf_test", "X-Sandbox-Token": "sbx_test"}


async def _upstream_app():
    seen: list[dict] = []

    async def status(request):
        seen.append(dict(request.headers))
        return web.json_response({"ok": True, "path": request.path, "q": request.query.get("q")})

    async def cmd(request):
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(f"data: {json.dumps({'echo': body})}\n\n".encode())
        await resp.write_eof()
        return resp

    async def ws(request):
        seen.append(dict(request.headers))
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        async for msg in sock:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await sock.send_str("echo:" + msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await sock.send_bytes(msg.data[::-1])
        return sock

    app = web.Application()
    app.router.add_get("/proxy/5000/status", status)
    app.router.add_post("/proxy/5000/cmd", cmd)
    app.router.add_get("/proxy/5000/ws", ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"127.0.0.1:{port}", seen


async def test_proxy_injects_headers_and_relays_http_sse_ws():
    up_runner, base, seen = await _upstream_app()
    proxy = _AuthProxy(f"https://{base}/proxy/5000", HEADERS)
    # the test upstream is plain http/ws, not TLS
    proxy._upstream = lambda path_qs, *, scheme: (  # type: ignore[method-assign]
        {"https://": "http://", "wss://": "ws://"}[scheme] + f"{base}/proxy/5000" + path_qs
    )
    await proxy.start()
    try:
        async with aiohttp.ClientSession() as cs:
            # plain GET with query string
            async with cs.get(f"{proxy.url}/status?q=1") as r:
                assert r.status == 200
                assert await r.json() == {"ok": True, "path": "/proxy/5000/status", "q": "1"}
            # streamed POST (SSE-style body)
            async with cs.post(f"{proxy.url}/cmd", json={"command": "echo hi"}) as r:
                assert r.status == 200
                text = await r.text()
                assert json.loads(text.split("data: ", 1)[1]) == {"echo": {"command": "echo hi"}}
            # websocket relay, both frame types
            async with cs.ws_connect(f"{proxy.url}/ws") as sock:
                await sock.send_str("ping")
                assert (await sock.receive()).data == "echo:ping"
                await sock.send_bytes(b"abc")
                assert (await sock.receive()).data == b"cba"
        # every upstream request carried the sandbox auth headers
        assert len(seen) == 2
        for h in seen:
            assert h["Authorization"] == HEADERS["Authorization"]
            assert h["X-Sandbox-Token"] == HEADERS["X-Sandbox-Token"]
    finally:
        await proxy.stop()
        await up_runner.cleanup()


@pytest.mark.parametrize("content_type", ["text/event-stream", "text/plain", "application/json"])
@pytest.mark.parametrize("body_relayed", [True, False])
async def test_client_close_after_relayed_result_is_not_a_warning(
    monkeypatch,
    caplog,
    content_type,
    body_relayed,
):
    async def chunks():
        yield b'data: {"success": true}\n\n'
        yield b": keepalive\n\n"

    upstream = SimpleNamespace(
        status=200,
        headers={},
        content_type=content_type,
        content=SimpleNamespace(iter_any=chunks),
    )
    context = AsyncMock()
    context.__aenter__.return_value = upstream
    proxy = _AuthProxy("https://example.hf.jobs", {})
    proxy.port = 12345
    proxy._client = SimpleNamespace(request=lambda *args, **kwargs: context)
    request = SimpleNamespace(
        host="127.0.0.1:12345",
        headers={},
        read=AsyncMock(return_value=b"{}"),
        rel_url=SimpleNamespace(raw_path_qs="/cmd"),
        method="POST",
        path="/cmd",
    )
    response = SimpleNamespace(
        headers={},
        prepared=True,
        prepare=AsyncMock(),
        write=AsyncMock(
            side_effect=[None, ConnectionResetError()] if body_relayed else ConnectionResetError()
        ),
        write_eof=AsyncMock(),
    )
    monkeypatch.setattr(web, "StreamResponse", lambda **kwargs: response)
    caplog.set_level(logging.DEBUG, logger="ale_run.environments.providers.hfsandbox")
    assert await proxy._handle(request) is response
    records = [record for record in caplog.records if "client closed" in record.message]
    assert len(records) == 1
    assert records[0].levelno == (logging.DEBUG if body_relayed else logging.WARNING)


def test_build_provider_config_defaults_and_overrides():
    cfg = _build_provider_config({})
    assert (cfg.image, cfg.flavor, cfg.idle_timeout) == ("ale-ubuntu22-docker", "cpu-xl", 3600)
    assert cfg.min_storage_gb == 200
    assert (cfg.transport, cfg.job_timeout) == ("job", "24h")
    assert cfg.resolution == (1024, 768)
    cfg = _build_provider_config(
        {"flavor": "cpu-basic", "idle_timeout": "2h", "resolution": [800, 600], "namespace": "org"}
    )
    assert (cfg.flavor, cfg.idle_timeout, cfg.resolution, cfg.namespace) == (
        "cpu-basic",
        "2h",
        (800, 600),
        "org",
    )
    assert _build_provider_config({"transport": "sandbox"}).transport == "sandbox"
    with pytest.raises(ValueError):
        _build_provider_config({"transport": "docker"})


def test_proxy_upstream_url_building():
    proxy = _AuthProxy("https://abc--5000.hf.jobs/", {})
    assert proxy._upstream("/cmd?x=1", scheme="https://") == "https://abc--5000.hf.jobs/cmd?x=1"
    assert proxy._upstream("/ws", scheme="wss://") == "wss://abc--5000.hf.jobs/ws"
    sub = _AuthProxy("https://h.hf.jobs/v1/proxy/5000", {})
    assert sub._upstream("/status", scheme="https://") == "https://h.hf.jobs/v1/proxy/5000/status"


def test_parse_volume_spec():
    v = parse_volume_spec("hf://buckets/ns/data:/mnt/data:ro")
    assert (v.type, v.source, v.mount_path, v.read_only, v.path) == (
        "bucket",
        "ns/data",
        "/mnt/data",
        True,
        None,
    )
    v = parse_volume_spec("hf://buckets/ns/data/sub/dir:/mnt/x")
    assert (v.read_only, v.path) == (None, "sub/dir")
    v = parse_volume_spec("hf://datasets/org/ds:/ds")
    assert (v.type, v.read_only) == ("dataset", True)
    for bad in (
        "buckets/ns/x:/m",
        "hf://buckets/ns/x",
        "hf://buckets/ns/x:/m:rx",
        "hf://buckets/onlyns:/m",
    ):
        with pytest.raises(ValueError):
            parse_volume_spec(bad)
    cfg = _build_provider_config({"volumes": ["hf://buckets/ns/data:/mnt/data:ro"]})
    assert cfg.volumes == ("hf://buckets/ns/data:/mnt/data:ro",)
