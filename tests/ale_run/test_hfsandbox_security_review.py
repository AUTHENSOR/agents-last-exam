from __future__ import annotations

import gzip

import aiohttp
from aiohttp import web
import pytest
from yarl import URL

from ale_run.environments.providers.hfsandbox import _AuthProxy


@pytest.fixture
async def proxy():
    seen = []

    async def upstream(request):
        seen.append(request)
        response = web.Response(
            body=gzip.compress(b"review-body"),
            headers={
                "Content-Encoding": "gzip",
                "Connection": "X-Internal",
                "X-Internal": "must-not-forward",
            },
        )
        response.headers.add("Set-Cookie", "first=1")
        response.headers.add("Set-Cookie", "second=2")
        return response

    application = web.Application()
    application.router.add_route("*", "/{tail:.*}", upstream)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    relay = _AuthProxy(f"https://127.0.0.1:{port}", {"Authorization": "Bearer review-placeholder"})
    relay._upstream = lambda path_qs, *, scheme: f"http://127.0.0.1:{port}{path_qs}"
    await relay.start()
    try:
        yield relay, seen
    finally:
        await relay.stop()
        await runner.cleanup()


@pytest.mark.parametrize(
    "headers",
    [{"Origin": "https://untrusted.example"}, {"Origin": "null"}, {"Host": "untrusted.example"}],
)
async def test_browser_and_rebinding_requests_are_rejected(proxy, headers):
    relay, seen = proxy
    async with aiohttp.ClientSession() as client:
        async with client.get(relay.url + "/status", headers=headers) as response:
            assert response.status == 403
    assert not seen


async def test_auth_override_is_case_insensitive_and_connection_headers_are_stripped(proxy):
    relay, seen = proxy
    async with aiohttp.ClientSession() as client:
        async with client.get(
            relay.url + "/status",
            headers={
                "authorization": "untrusted",
                "Connection": "X-Private",
                "X-Private": "must-not-forward",
            },
        ) as response:
            assert response.status == 200
            assert response.headers.getall("Set-Cookie") == ["first=1", "second=2"]
            assert "X-Internal" not in response.headers
            assert await response.read() == b"review-body"
    assert seen[0].headers.getall("Authorization") == ["Bearer review-placeholder"]
    assert "X-Private" not in seen[0].headers


async def test_encoded_paths_and_queries_are_not_decoded_by_proxy(proxy):
    relay, seen = proxy
    path = "/file%2Fname?q=one%2Ftwo&literal=%252F"
    async with aiohttp.ClientSession() as client:
        async with client.get(URL(relay.url + path, encoded=True)) as response:
            assert response.status == 200
            await response.read()
    assert seen[0].raw_path == path


async def test_startup_failure_closes_thread_and_client(monkeypatch):
    relay = _AuthProxy("https://review.hf.jobs", {})

    async def fail(instance):
        instance._client = aiohttp.ClientSession()
        raise OSError("review startup failure")

    monkeypatch.setattr(_AuthProxy, "_start_on_loop", fail)
    try:
        with pytest.raises(OSError):
            await relay.start()
        assert relay._thread is None
        assert relay._client is None
    finally:
        await relay.stop()
