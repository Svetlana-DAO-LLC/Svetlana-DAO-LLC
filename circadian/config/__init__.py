"""CircAIdian configuration"""
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CircadianConfig:
    soul_path: str = "/home/hermes/.hermes/SOUL.md"
    memories_path: str = "/home/hermes/.hermes/memories"
    bank_path: str = "/home/hermes/repos/jr-artifacts/hermes-bank/bank"
    corrections_db_path: str = "/home/hermes/.hermes/circadian_corrections.db"
    dream_log_path: str = "/home/hermes/.hermes/circadian_dreams.log"
    nudge_log_path: str = "/home/hermes/.hermes/circadian_nudges.log"
    correction_batch_interval: int = 60
    idle_check_interval: int = 30
    dream_cycle_interval: int = 300
    heartbeat_interval: int = 60
    correction_confidence_threshold: float = 0.7
    idle_short_threshold: int = 300
    idle_long_threshold: int = 1800
    max_context_tokens: int = 16000
    timezone: str = "Europe/Berlin"
    sleep_hours_start: int = 23
    sleep_hours_end: int = 7
    opencode_model: str = "glm-5.1"
    opencode_endpoint: str = None
    api_host: str = "127.0.0.1"
    api_port: int = 9378

    @classmethod
    def from_env(cls) -> "CircadianConfig":
        return cls(
            soul_path=os.getenv("CIRCADIAN_SOUL_PATH", "/home/hermes/.hermes/SOUL.md"),
            memories_path=os.getenv("CIRCADIAN_MEMORIES_PATH", "/home/hermes/.hermes/memories"),
            bank_path=os.getenv("CIRCADIAN_BANK_PATH", "/home/hermes/repos/jr-artifacts/hermes-bank/bank"),
            corrections_db_path=os.getenv("CIRCADIAN_DB_PATH", "/home/hermes/.hermes/circadian_corrections.db"),
            api_port=int(os.getenv("CIRCADIAN_API_PORT", "9378")),
            opencode_endpoint=os.getenv("OPENCODE_ENDPOINT"),
        )

config = CircadianConfig.from_env()
