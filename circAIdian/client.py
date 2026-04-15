"""
CircAIdian client library — for main agent integration.

Usage:
    from circAIdian.client import CircadianClient

    async with CircadianClient() as client:
        await client.observe("session_1", "user message", "agent response")
        nudge = await client.get_nudge()
"""

import os
from dataclasses import dataclass
from typing import Optional, List

import aiohttp


@dataclass
class Nudge:
    id: str
    content: str
    category: str
    priority: str
    confidence: float
    created_at: str


@dataclass
class Correction:
    id: int
    wrong_claim: str
    correction: str
    source: str
    confidence: float
    applied: bool
    timestamp: str


class CircadianClient:
    """Async client for CircAIdian daemon.

    Set CIRCADIAN_API_TOKEN env var if the daemon requires auth.
    Set CIRCADIAN_API_HOST and CIRCADIAN_API_PORT to override defaults.
    """

    def __init__(
        self,
        base_url: str = None,
        token: str = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url or f"http://{os.getenv('CIRCADIAN_API_HOST', '127.0.0.1')}:{os.getenv('CIRCADIAN_API_PORT', '9378')}"
        self.token = token or os.getenv("CIRCADIAN_API_TOKEN", "")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        headers = {}
        if self.token:
            headers["X-Auth-Token"] = self.token
        self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def _post(self, path: str, json: dict = None) -> dict:
        async with self._session.post(f"{self.base_url}{path}", json=json, timeout=self.timeout) as resp:
            data = await resp.json()
            if not data.get("success"):
                raise CircadianError(data.get("error", f"API error on {path}"))
            return data["data"]

    async def _get(self, path: str) -> dict:
        async with self._session.get(f"{self.base_url}{path}", timeout=self.timeout) as resp:
            data = await resp.json()
            if not data.get("success"):
                raise CircadianError(data.get("error", f"API error on {path}"))
            return data["data"]

    # ── Core ────────────────────────────────────────────────────────────────

    async def observe(self, session_id: str, user_msg: str, agent_output: str = "") -> bool:
        """Send a conversation turn for processing."""
        result = await self._post("/observe", {
            "session_id": session_id,
            "user_msg": user_msg,
            "agent_output": agent_output,
        })
        return result.get("status") == "observed"

    async def correct(self, wrong_claim: str, correction: str, source: str = "user") -> bool:
        """Manually register a correction."""
        result = await self._post("/correction", {
            "wrong_claim": wrong_claim,
            "correction": correction,
            "source": source,
        })
        return result.get("queued", False)

    async def preference(self, preference: str) -> bool:
        """Register a user preference."""
        result = await self._post("/preference", {"preference": preference})
        return result is not None

    async def set_task(self, task: str) -> bool:
        """Inform the daemon of the current task context."""
        result = await self._post("/task", {"task": task})
        return result is not None

    # ── Read ────────────────────────────────────────────────────────────────

    async def get_nudge(self) -> Optional[Nudge]:
        """Poll for the next nudge. Returns None if queue is empty."""
        result = await self._get("/nudge")
        nudge_data = result.get("nudge")
        if not nudge_data:
            return None
        return Nudge(**nudge_data)

    async def get_state(self) -> dict:
        """Get the current emotional/interactional state."""
        return await self._get("/state")

    async def get_context_stats(self) -> dict:
        """Get context manager statistics."""
        result = await self._get("/context")
        return result.get("stats", {})

    async def get_corrections(self, limit: int = 20, applied: bool = None) -> List[Correction]:
        """List recent corrections. Pass applied=True/False to filter."""
        params = f"?limit={limit}"
        if applied is not None:
            params += f"&applied={int(applied)}"
        result = await self._get(f"/corrections{params}")
        return [Correction(**c) for c in result.get("corrections", [])]

    async def undo_correction(self, correction_id: int) -> dict:
        """Delete a correction by ID."""
        async with self._session.delete(
            f"{self.base_url}/corrections/{correction_id}", timeout=self.timeout
        ) as resp:
            data = await resp.json()
            if not data.get("success"):
                raise CircadianError(data.get("error", f"Cannot delete correction {correction_id}"))
            return data["data"]

    async def health(self) -> dict:
        """Daemon health check."""
        return await self._get("/health")

    async def info(self) -> dict:
        """Daemon version and status."""
        return await self._get("/")


class CircadianError(Exception):
    """CircAIdian API error."""
    pass
