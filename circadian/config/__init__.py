"""CircAIdian configuration — single source of truth"""
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict


class IdleState(Enum):
    ACTIVE = "active"
    IDLE_SHORT = "idle_short"
    IDLE_LONG = "idle_long"
    SLEEPING = "sleeping"
    FOCUS = "focus"


@dataclass
class CircadianConfig:
    # Paths
    soul_path: str = "/home/hermes/.hermes/SOUL.md"
    memories_path: str = "/home/hermes/.hermes/memories"
    bank_path: str = "/home/hermes/repos/jr-artifacts/hermes-bank/bank"
    corrections_db_path: str = "/home/hermes/.hermes/circadian_corrections.db"
    dream_log_path: str = "/home/hermes/.hermes/circadian_dreams.log"
    nudge_log_path: str = "/home/hermes/.hermes/circadian_nudges.log"
    # Intervals (seconds)
    correction_batch_interval: int = 60
    idle_check_interval: int = 30
    dream_cycle_interval: int = 300
    heartbeat_interval: int = 60
    # Correction
    correction_confidence_threshold: float = 0.7
    # Idle detection
    idle_short_threshold: int = 300
    idle_long_threshold: int = 1800
    idle_thresholds: tuple = field(default_factory=lambda: (300, 1800))
    # Context
    max_context_tokens: int = 16000
    # Circadian rhythm
    timezone: str = "Europe/Berlin"
    sleep_hours_start: int = 23
    sleep_hours_end: int = 7
    dream_intensity: Dict[IdleState, int] = field(default_factory=lambda: {
        IdleState.ACTIVE: 0,
        IdleState.IDLE_SHORT: 1,
        IdleState.IDLE_LONG: 3,
        IdleState.SLEEPING: 10,
        IdleState.FOCUS: 0,
    })
    # OpenCode (dream engine)
    opencode_model: str = "glm-5.1"
    opencode_endpoint: str = None
    # API
    api_host: str = "127.0.0.1"
    api_port: int = 9378
    # Webhook (optional push-based nudge delivery)
    webhook_url: str = None

    @classmethod
    def from_env(cls) -> "CircadianConfig":
        """Build config from environment variables."""
        idle_short = int(os.getenv("CIRCADIAN_IDLE_SHORT", "300"))
        idle_long = int(os.getenv("CIRCADIAN_IDLE_LONG", "1800"))
        return cls(
            soul_path=os.getenv("CIRCADIAN_SOUL_PATH", "/home/hermes/.hermes/SOUL.md"),
            memories_path=os.getenv("CIRCADIAN_MEMORIES_PATH", "/home/hermes/.hermes/memories"),
            bank_path=os.getenv("CIRCADIAN_BANK_PATH", "/home/hermes/repos/jr-artifacts/hermes-bank/bank"),
            corrections_db_path=os.getenv("CIRCADIAN_DB_PATH", "/home/hermes/.hermes/circadian_corrections.db"),
            dream_log_path=os.getenv("CIRCADIAN_DREAM_LOG", "/home/hermes/.hermes/circadian_dreams.log"),
            nudge_log_path=os.getenv("CIRCADIAN_NUDGE_LOG", "/home/hermes/.hermes/circadian_nudges.log"),
            correction_batch_interval=int(os.getenv("CIRCADIAN_BATCH_INTERVAL", "60")),
            idle_check_interval=int(os.getenv("CIRCADIAN_IDLE_CHECK_INTERVAL", "30")),
            dream_cycle_interval=int(os.getenv("CIRCADIAN_DREAM_INTERVAL", "300")),
            heartbeat_interval=int(os.getenv("CIRCADIAN_HEARTBEAT_INTERVAL", "60")),
            idle_short_threshold=idle_short,
            idle_long_threshold=idle_long,
            idle_thresholds=(idle_short, idle_long),
            max_context_tokens=int(os.getenv("CIRCADIAN_MAX_TOKENS", "16000")),
            api_host=os.getenv("CIRCADIAN_API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("CIRCADIAN_API_PORT", "9378")),
            opencode_model=os.getenv("CIRCADIAN_OPENCODE_MODEL", "glm-5.1"),
            opencode_endpoint=os.getenv("OPENCODE_ENDPOINT"),
            webhook_url=os.getenv("CIRCADIAN_WEBHOOK_URL") or None,
        )


config = CircadianConfig.from_env()
