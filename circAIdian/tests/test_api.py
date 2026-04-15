"""Tests for CircadianAPI using aiohttp test client."""
import tempfile
from pathlib import Path

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import TestClient, TestServer

from daemon import CircadianAPI, CorrectionHandler, NudgeEngine, EmotionalState
from daemon import ActiveContextManager


@pytest.fixture(autouse=True)
def no_auth():
    """Disable auth token for all API tests."""
    import os
    from daemon import api as api_module
    saved = os.environ.pop("CIRCADIAN_API_TOKEN", None)
    api_module._API_TOKEN = None  # bypass module-level token
    yield
    if saved is not None:
        os.environ["CIRCADIAN_API_TOKEN"] = saved
        api_module._API_TOKEN = saved


@pytest.fixture
def temp_api_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield {"soul": tmp / "SOUL.md", "db": tmp / "corrections.db"}


async def make_api(temp_api_paths):
    ch = CorrectionHandler(
        soul_path=str(temp_api_paths["soul"]),
        db_path=str(temp_api_paths["db"]),
    )
    ne = NudgeEngine()
    cm = ActiveContextManager()
    es = EmotionalState()
    return CircadianAPI(ch, ne, cm, es, host="127.0.0.1", port=0)


async def start_api(api):
    """Start API and return (TestServer, TestClient) tuple."""
    await api.start()
    ts = TestServer(api.app)
    client = TestClient(ts)
    await client.start_server()
    return client, api


class TestCircadianAPI:
    @pytest.mark.asyncio
    async def test_health(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
            assert data["data"]["status"] == "healthy"
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_root(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.get("/")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"]["daemon"] == "CircAIdian"
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_observe(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.post("/observe", json={
                "session_id": "s1", "user_msg": "hello", "agent_output": "hi"
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_correction_requires_fields(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.post("/correction", json={})
            assert resp.status == 400
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_correction_queues(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.post("/correction", json={
                "wrong_claim": "wrong", "correction": "correct"
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_nudge_empty(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.get("/nudge")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"]["nudge"] is None
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_state(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.get("/state")
            assert resp.status == 200
            data = await resp.json()
            assert "valence" in data["data"]
            assert "energy" in data["data"]
        finally:
            await client.close()
            await api.stop()

    @pytest.mark.asyncio
    async def test_context_stats(self, temp_api_paths):
        api = await make_api(temp_api_paths)
        client, api = await start_api(api)
        try:
            resp = await client.get("/context")
            assert resp.status == 200
            data = await resp.json()
            assert "stats" in data["data"]
        finally:
            await client.close()
            await api.stop()
