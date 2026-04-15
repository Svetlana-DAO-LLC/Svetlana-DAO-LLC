#!/usr/bin/env python3
"""CircAIdian daemon entry point — production-ready"""
import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("circadian")

sys.path.insert(0, str(Path(__file__).parent))

from daemon import (
    CorrectionHandler,
    EmotionalState,
    NudgeEngine,
    ActiveContextManager,
    CircadianAPI,
)
from config import config, CircadianConfig


class DreamEngine:
    """Subconscious processing via OpenCode (GLM-5.1)."""

    def __init__(self, cfg: CircadianConfig, context_manager: ActiveContextManager,
                 nudge_engine: NudgeEngine, emotional_state: EmotionalState):
        self.cfg = cfg
        self.context_manager = context_manager
        self.nudge_engine = nudge_engine
        self.emotional_state = emotional_state
        self._log_path = Path(cfg.dream_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_cycle(self, intensity: int) -> bool:
        """Run one dream cycle. Returns True if an insight was generated."""
        if intensity <= 0:
            return False

        prompt = self._build_dream_prompt(intensity)
        result = await self._call_opencode(prompt)
        if not result:
            return False

        insight = self._parse_insight(result)
        if insight:
            self._log_dream(insight, result)
            await self.nudge_engine.enqueue_dream_insight(insight, confidence=0.3)
            return True
        return False

    def _build_dream_prompt(self, intensity: int) -> str:
        context_snippet = self.context_manager.get_context_for_prompt()
        dream_depth = {
            1: "brief impression",
            3: "moderate reflection",
            10: "deep consolidation",
        }.get(intensity, "reflection")

        return f"""You are CircAIdian, the subconscious of an AI agent.
A {dream_depth} cycle is occurring. Review the recent context below and
generate a single concise insight or pattern that might help the main agent.

Recent context:
{context_snippet[:2000]}

Respond with ONLY a JSON object: {{"insight": "<one sentence insight>", "pattern": "<optional pattern name>"}}
Do not include any other text."""

    async def _call_opencode(self, prompt: str) -> str:
        """Call OpenCode CLI with the dream prompt."""
        endpoint_arg = []
        if self.cfg.opencode_endpoint:
            endpoint_arg = ["--endpoint", self.cfg.opencode_endpoint]

        try:
            result = await asyncio.create_subprocess_exec(
                "opencode", "chat", *endpoint_arg,
                "--model", self.cfg.opencode_model,
                "--json",
                input=prompt.encode(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate(timeout=60)
            if result.returncode != 0:
                logger.warning(f"opencode failed: {stderr.decode()[:200]}")
                return ""
            return stdout.decode().strip()
        except FileNotFoundError:
            logger.debug("opencode CLI not found — dream engine skipped")
            return ""
        except asyncio.TimeoutError:
            logger.warning("opencode timed out after 60s")
            return ""
        except Exception as e:
            logger.warning(f"Dream engine error: {e}")
            return ""

    def _parse_insight(self, raw: str) -> str:
        """Extract insight text from OpenCode JSON response."""
        try:
            # Try direct JSON parse
            data = json.loads(raw)
            return data.get("insight", "")
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
            try:
                data = json.loads(raw.strip())
                return data.get("insight", "")
            except json.JSONDecodeError:
                pass
        # Fallback: return raw as insight if non-empty
        if raw.strip():
            return raw.strip()[:200]
        return ""

    def _log_dream(self, insight: str, raw_response: str):
        """Append dream to log file."""
        timestamp = datetime.now().isoformat()
        with open(self._log_path, "a") as f:
            f.write(f"[{timestamp}] INTENSITY={self.emotional_state.idle_state.value} | {insight}\n")


class CircAIdianDaemon:
    def __init__(self, cfg: CircadianConfig):
        self.config = cfg
        self.correction_handler = CorrectionHandler(
            soul_path=cfg.soul_path,
            db_path=cfg.corrections_db_path,
            batch_interval=cfg.correction_batch_interval,
        )
        self.emotional_state = EmotionalState()
        self.nudge_engine = NudgeEngine()
        self.context_manager = ActiveContextManager(max_context_tokens=cfg.max_context_tokens)
        self.dream_engine = DreamEngine(cfg, self.context_manager, self.nudge_engine, self.emotional_state)
        self.api = CircadianAPI(
            correction_handler=self.correction_handler,
            nudge_engine=self.nudge_engine,
            context_manager=self.context_manager,
            emotional_state=self.emotional_state,
            host=cfg.api_host,
            port=cfg.api_port,
        )
        self._tasks = []
        self._running = False

    async def start(self):
        logger.info("Starting CircAIdian daemon...")
        await self.api.start()
        self._running = True
        self._tasks.append(asyncio.create_task(self._correction_batch_loop()))
        self._tasks.append(asyncio.create_task(self._idle_check_loop()))
        self._tasks.append(asyncio.create_task(self._dream_cycle_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        logger.info("CircAIdian daemon started successfully")
        while self._running:
            await asyncio.sleep(10)

    async def stop(self):
        logger.info("Stopping CircAIdian daemon...")
        self._running = False
        for task in self._tasks:
            task.cancel()
        await self.api.stop()

    async def _correction_batch_loop(self):
        while self._running:
            await asyncio.sleep(self.config.correction_batch_interval)
            try:
                applied = await self.correction_handler.process_batch()
                for correction in applied:
                    self.emotional_state.corrections_processed_today += 1
                    await self.nudge_engine.enqueue_correction_notification(
                        correction.wrong_claim, correction.correction
                    )
                # Proactive webhook delivery
                if self.config.webhook_url and applied:
                    await self._deliver_webhook(applied)
            except Exception as e:
                logger.error(f"Error in correction batch loop: {e}")

    async def _idle_check_loop(self):
        while self._running:
            await asyncio.sleep(self.config.idle_check_interval)
            try:
                self.emotional_state.idle_state = self.emotional_state.compute_idle_state(
                    self.config.idle_thresholds
                )
            except Exception as e:
                logger.error(f"Error in idle check loop: {e}")

    async def _dream_cycle_loop(self):
        while self._running:
            await asyncio.sleep(self.config.dream_cycle_interval)
            try:
                intensity = self.config.dream_intensity.get(self.emotional_state.idle_state, 0)
                if intensity > 0:
                    self.emotional_state.dream_count_today += intensity
                    self.emotional_state.last_dream_cycle = datetime.now()
                    insight_generated = await self.dream_engine.run_cycle(intensity)
                    if insight_generated:
                        self.emotional_state.nudge_count_today += 1
            except Exception as e:
                logger.error(f"Error in dream cycle loop: {e}")

    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval)
            logger.debug("Heartbeat OK")

    async def _deliver_webhook(self, corrections: list):
        """POST applied corrections to the configured webhook URL."""
        import aiohttp
        try:
            payload = {
                "event": "corrections_applied",
                "count": len(corrections),
                "corrections": [
                    {"wrong": c.wrong_claim, "correct": c.correction, "source": c.source}
                    for c in corrections
                ],
                "timestamp": datetime.now().isoformat(),
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook POST failed: {resp.status}")
                    else:
                        logger.info(f"Webhook delivered {len(corrections)} correction(s)")
        except Exception as e:
            logger.warning(f"Webhook delivery error: {e}")


async def run_once(cfg: CircadianConfig):
    """Single-cycle dry run to verify correction detection."""
    logger.info("Running CircAIdian in single-cycle mode...")
    handler = CorrectionHandler(cfg.soul_path, cfg.corrections_db_path, batch_interval=1)
    test_texts = [
        "No, it's Claude Code that's wrong",
        "Actually it's GLM-5.1, not GPT-4",
        "Wait, I meant to say you use OpenCode not Codex",
    ]
    for text in test_texts:
        correction = handler._detect_correction(text)
        if correction:
            logger.info(f"Detected: {correction.wrong_claim} -> {correction.correction} "
                        f"(confidence: {correction.confidence})")
        else:
            logger.info(f"No correction detected in: {text}")


def check_running(cfg: CircadianConfig) -> bool:
    """Check if daemon is running on the configured host:port. Uses context manager."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((cfg.api_host, cfg.api_port))
            return result == 0
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="CircAIdian daemon")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--check", action="store_true", help="Check if daemon is running")
    args = parser.parse_args()

    if args.check:
        running = check_running(config)
        print("CircAIdian daemon is RUNNING" if running else "CircAIdian daemon is NOT running")
        sys.exit(0 if running else 1)

    if args.once:
        asyncio.run(run_once(config))
        sys.exit(0)

    daemon = CircAIdianDaemon(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        loop.create_task(daemon.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(daemon.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
