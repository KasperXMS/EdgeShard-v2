"""VLLMClient unit tests against a local OpenAI-compatible fake (spec 22.1).

A small in-process HTTP server plays vLLM's OpenAI-compatible API, so the
master's vLLM test requests are exercised on the CPU development host
without Docker, GPU, or the real vLLM image.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from edgeshard.control.mock.client import VLLMClient

MODEL = "/models/tiny-llama"


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    server_version = "FakeVLLM/1.0"

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._respond(200, {"data": [{"id": MODEL}]})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.server.requests.append((self.path, body))
        if self.path == "/v1/completions":
            if body.get("model") != MODEL:
                self._respond(404, {"error": "unknown model"})
                return
            self._respond(
                200,
                {"choices": [{"text": "hello tokens", "finish_reason": "length"}]},
            )
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def fake_vllm_server() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def endpoint_of(server: ThreadingHTTPServer) -> str:
    return f"127.0.0.1:{server.server_address[1]}"


async def test_list_models(fake_vllm_server: ThreadingHTTPServer) -> None:
    async with VLLMClient(endpoint_of(fake_vllm_server), model=MODEL) as client:
        assert await client.list_models() == [MODEL]


async def test_complete_sends_deterministic_greedy_request(
    fake_vllm_server: ThreadingHTTPServer,
) -> None:
    async with VLLMClient(endpoint_of(fake_vllm_server), model=MODEL) as client:
        completion = await client.complete([3, 7, 11, 19], max_tokens=5)

    assert completion.text == "hello tokens"
    assert completion.finish_reason == "length"
    (path, body), = fake_vllm_server.requests
    assert path == "/v1/completions"
    assert body == {
        "model": MODEL,
        "prompt": [3, 7, 11, 19],
        "max_tokens": 5,
        "temperature": 0.0,  # deterministic greedy, like the rest of Phase 0
    }


async def test_complete_rejects_bad_max_tokens(
    fake_vllm_server: ThreadingHTTPServer,
) -> None:
    async with VLLMClient(endpoint_of(fake_vllm_server), model=MODEL) as client:
        with pytest.raises(ValueError, match="max_tokens"):
            await client.complete([1, 2], max_tokens=0)


async def test_http_errors_propagate(fake_vllm_server: ThreadingHTTPServer) -> None:
    async with VLLMClient(endpoint_of(fake_vllm_server), model="wrong-model") as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete([1, 2], max_tokens=1)
