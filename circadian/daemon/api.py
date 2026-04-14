"""REST API for CircAIdian daemon"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

# Auth token — set via CIRCADIAN_API_TOKEN env var. If unset, auth is disabled.
_API_TOKEN = os.getenv("CIRCADIAN_API_TOKEN")
_ALLOWED_SOUL_DIR = Path("/home/hermes/.hermes").resolve()


class ApiError(Exception):
    """Structured API error with HTTP status."""
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def _validate_soul_path(env_var_path: str) -> Path:
    """Resolve soul path from env var and validate it's within allowed directory."""
    raw = Path(env_var_path).expanduser().resolve()
    if ".." in env_var_path:
        raise ApiError(400, "Invalid soul path: '..' not allowed")
    if not str(raw).startswith(str(_ALLOWED_SOUL_DIR)):
        raise ApiError(400, "Invalid soul path: must be within /home/hermes/.hermes")
    return raw


def _json_response(success: bool, data=None, error: str = None) -> dict:
    """Standardized JSON response envelope."""
    return {"success": success, "data": data or {}, "error": error}


async def _auth_middleware(app, handler):
    """Auth middleware: checks X-Auth-Token if CIRCADIAN_API_TOKEN is set."""
    async def middleware(request):
        if _API_TOKEN:
            token = request.headers.get("X-Auth-Token", "")
            if token != _API_TOKEN:
                logger.warning(f"Unauthorized request to {request.path} from {request.remote}")
                return web.json_response(
                    _json_response(False, error="Unauthorized"),
                    status=401
                )
        return await handler(request)
    return middleware


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
        self.app = web.Application(middlewares=[_auth_middleware] if _API_TOKEN else [])
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

    async def _read_soul_content(self) -> str:
        """Read SOUL.md with path traversal protection."""
        env_path = os.getenv("CIRCADIAN_SOUL_PATH", "/home/hermes/.hermes/SOUL.md")
        soul_path = _validate_soul_path(env_path)
        return soul_path.read_text() if soul_path.exists() else ""

    async def handle_observe(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            soul_content = await self._read_soul_content()
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
            return web.json_response(_json_response(True, {"status": "observed"}))
        except ApiError as e:
            logger.error(f"ApiError in /observe: {e.message}")
            return web.json_response(_json_response(False, error=e.message), status=e.status)
        except Exception as e:
            logger.exception(f"Error in /observe: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_correction(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            wrong_claim = body.get("wrong_claim", body.get("wrong", ""))
            correction_text = body.get("correction", body.get("correct", ""))
            source = body.get("source", "user")
            if not wrong_claim or not correction_text:
                return web.json_response(
                    _json_response(False, error="Both 'wrong_claim' and 'correction' are required"),
                    status=400
                )
            self.context_manager.add_correction(wrong_claim, correction_text, source)
            return web.json_response(_json_response(True, {"queued": True}))
        except ApiError as e:
            logger.error(f"ApiError in /correction: {e.message}")
            return web.json_response(_json_response(False, error=e.message), status=e.status)
        except Exception as e:
            logger.exception(f"Error in /correction: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_preference(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            preference = body.get("preference", "")
            if not preference:
                return web.json_response(
                    _json_response(False, error="'preference' field is required"),
                    status=400
                )
            self.context_manager.add_user_preference(preference)
            return web.json_response(_json_response(True))
        except ApiError as e:
            logger.error(f"ApiError in /preference: {e.message}")
            return web.json_response(_json_response(False, error=e.message), status=e.status)
        except Exception as e:
            logger.exception(f"Error in /preference: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_task(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            task = body.get("task", "")
            self.context_manager.set_current_task(task)
            return web.json_response(_json_response(True))
        except ApiError as e:
            logger.error(f"ApiError in /task: {e.message}")
            return web.json_response(_json_response(False, error=e.message), status=e.status)
        except Exception as e:
            logger.exception(f"Error in /task: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_nudge(self, request: web.Request) -> web.Response:
        try:
            nudge = await self.nudge_engine.get_next()
            if nudge:
                return web.json_response(_json_response(True, {
                    "nudge": {
                        "id": nudge.nudge_id,
                        "content": nudge.content,
                        "category": nudge.category.value,
                        "priority": nudge.priority.name,
                        "confidence": nudge.confidence,
                        "created_at": nudge.created_at.isoformat(),
                    }
                }))
            return web.json_response(_json_response(True, {"nudge": None}))
        except Exception as e:
            logger.exception(f"Error in /nudge: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_state(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(_json_response(True, self.emotional_state.to_dict()))
        except Exception as e:
            logger.exception(f"Error in /state: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_context(self, request: web.Request) -> web.Response:
        try:
            stats = self.context_manager.get_stats()
            return web.json_response(_json_response(True, {"stats": stats}))
        except Exception as e:
            logger.exception(f"Error in /context: {e}")
            return web.json_response(_json_response(False, error="Internal server error"), status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(_json_response(True, {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }))
