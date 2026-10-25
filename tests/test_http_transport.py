"""`--transport streamable-http`: one server, several clients, one copy of each model.

Spawns a real server on a free loopback port. Only PP-OCRv6 (about 0.5 GB, ~1 s) is loaded.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

FIXTURE = str(Path(__file__).parent / "layout_two_column.png")


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def http_server() -> Iterator[str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "fusion_vision_mcp", "--transport", "streamable-http", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited with {proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("server never started listening")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _call(url: str, tool: str, arguments: dict[str, object]) -> str:
    async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        res = await session.call_tool(tool, arguments)
        return "\n".join(cast(TextContent, c).text for c in res.content)


@pytest.mark.anyio
async def test_two_clients_share_one_loaded_model(http_server: str) -> None:
    text = await _call(http_server, "ocr", {"src": FIXTURE})
    assert "Attendee" in text

    # A second, separate session sees the model the first one loaded: one process, one copy.
    status = json.loads(await _call(http_server, "server_status", {}))
    loaded = {m["model"] for m in status["models"] if m["loaded"]}
    assert "PP-OCRv6" in loaded
    assert [e["model"] for e in status["recent_events"] if e["event"] == "loaded"].count("PP-OCRv6") == 1


def test_a_foreign_host_header_is_refused(http_server: str) -> None:
    """DNS-rebinding protection: a page that resolves evil.example to 127.0.0.1 still sends its own Host."""
    request = urllib.request.Request(
        http_server,
        data=b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}',
        headers={"Host": "evil.example", "Content-Type": "application/json", "Accept": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)

    assert 400 <= excinfo.value.code < 500


def test_a_non_loopback_host_is_warned_about() -> None:
    """Only the warning is checked. 192.0.2.1 is a documentation address no machine has, so the bind
    fails and the server exits without ever listening (0.0.0.0 would raise a firewall prompt)."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fusion_vision_mcp",
            "--transport",
            "streamable-http",
            "--host",
            "192.0.2.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        _, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert "no authentication" in " ".join(err.split())
