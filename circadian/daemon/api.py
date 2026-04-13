"""REST API for CircAIdian daemon"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

class CircadianAPI:
    def __init__(self, correction_handler, nudge_engine, context_manager, emotional_state,
                 host: str = "127.0.0.1", port: int = 9378):
        self.correction_handler = correction_handler
        self.nudge_engine = nudge_engine
        self.context_manager = context_manager
        self.emotional_state = emotional_state
        self.host = host
        self.port = port
        self.app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
    
    async def start(self):
        self.app = web.Application()
        self.app.router.add_post("/observe", self.handle_observe)
        self.app.router.add_post("/correction", self.handle_correction)
        self.app.router.add_post("/preference", self.handle_preference)
        self.app.router.add_post("/task", self.handle_task)
        self.app.router.add_get("/nudge", self.handle_nudge)
        self.app.router.add_get("/state", self.handle_state)
        self.app.router.add_get("/context", self.handle_context)
        self.app.router.add_get("/health", self.handle_health)
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"CircAIdian API started at http://{self.host}:{self.port}")
    
    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
    
    async def handle_observe(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            import os
            soul_path = Path(os.getenv("CIRCADIAN_SOUL_PATH", "/home/hermes/.hermes/SOUL.md"))
            soul_content = soul_path.read_text() if soul_path.exists() else ""
            await self.correction_handler.process_observation(
                session_id=body.get("session_id", "default"),
                user_msg=body.get("user_msg", ""),
                agent_output=body.get("agent_output", ""),
                soul_content=soul_content,
            )
            if body.get("user_msg") or body.get("agent_output"):
                self.context_manager.add_message_pair(
                    body.get("user_msg", ""), body.get("agent_output", "")
                )
            if body.get("user_msg"):
                self.emotional_state.last_user_activity = datetime.now()
                self.emotional_state.idle_state = self.emotional_state.compute_idle_state()
            return web.Response(json={"status": "ok"})
        except Exception as e:
            logger.error(f"Error in /observe: {e}")
            return web.Response(status=500, text=str(e))
    
    async def handle_correction(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            self.context_manager.add_correction(
                body.get("wrong", ""), body.get("correct", ""), body.get("source", "user")
            )
            return web.Response(json={"status": "ok", "queued": True})
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    async def handle_preference(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            self.context_manager.add_user_preference(body.get("preference", ""))
            return web.Response(json={"status": "ok"})
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    async def handle_task(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            self.context_manager.set_current_task(body.get("task", ""))
            return web.Response(json={"status": "ok"})
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    async def handle_nudge(self, request: web.Request) -> web.Response:
        nudge = await self.nudge_engine.get_next()
        if nudge:
            return web.Response(json={
                "nudge": {
                    "id": nudge.nudge_id, "content": nudge.content,
                    "category": nudge.category.value, "priority": nudge.priority.value,
                    "confidence": nudge.confidence, "created_at": nudge.created_at.isoformat(),
                }
            })
        return web.Response(json={"nudge": None})
    
    async def handle_state(self, request: web.Request) -> web.Response:
        return web.Response(json=self.emotional_state.to_dict())
    
    async def handle_context(self, request: web.Request) -> web.Response:
        stats = self.context_manager.get_stats()
        return web.Response(json={"stats": stats})
    
    async def handle_health(self, request: web.Request) -> web.Response:
        return web.Response(json={"status": "healthy", "timestamp": datetime.now().isoformat()})
