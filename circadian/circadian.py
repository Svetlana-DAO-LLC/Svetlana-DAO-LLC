#!/usr/bin/env python3
"""CircAIdian daemon entry point"""
import argparse
import asyncio
import logging
import os
import signal
import sys
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
    CircadianConfig,
    NudgeEngine,
    ActiveContextManager,
    CircadianAPI,
)
from config import config

class CircAIdianDaemon:
    def __init__(self, cfg):
        self.config = cfg
        self.correction_handler = CorrectionHandler(
            soul_path=cfg.soul_path,
            db_path=cfg.corrections_db_path,
            batch_interval=cfg.correction_batch_interval,
        )
        self.emotional_state = EmotionalState()
        self.nudge_engine = NudgeEngine()
        self.context_manager = ActiveContextManager(max_context_tokens=cfg.max_context_tokens)
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
                    if intensity >= 3:
                        await self.nudge_engine.enqueue_dream_insight(
                            "Dream cycle complete. Processed recent memories.", confidence=0.3
                        )
            except Exception as e:
                logger.error(f"Error in dream cycle loop: {e}")
    
    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval)
            logger.debug("Heartbeat OK")


async def run_once(cfg):
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
            logger.info(f"Detected: {correction.wrong_claim} -> {correction.correction} (confidence: {correction.confidence})")
        else:
            logger.info(f"No correction detected in: {text}")


def check_running(cfg):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((cfg.api_host, cfg.api_port))
    sock.close()
    return result == 0


def main():
    parser = argparse.ArgumentParser(description="CircAIdian daemon")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--check", action="store_true", help="Check if daemon is running")
    args = parser.parse_args()
    
    if args.check:
        print("CircAIdian daemon is RUNNING" if check_running(config) else "CircAIdian daemon is NOT running")
        sys.exit(0 if check_running(config) else 1)
    
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
