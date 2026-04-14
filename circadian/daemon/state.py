"""Emotional state and circadian config for CircAIdian"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from zoneinfo import ZoneInfo

# Default sleep hours — configurable via CircadianConfig
DEFAULT_SLEEP_START = 23
DEFAULT_SLEEP_END = 7

class IdleState(Enum):
    ACTIVE = "active"
    IDLE_SHORT = "idle_short"
    IDLE_LONG = "idle_long"
    SLEEPING = "sleeping"
    FOCUS = "focus"

class EmotionalValence(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"

@dataclass
class EmotionalState:
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    energy: float = 0.5
    arousal: float = 0.3
    dominance: float = 0.5
    last_user_activity: datetime = field(default_factory=datetime.now)
    last_dream_cycle: Optional[datetime] = None
    dream_count_today: int = 0
    nudge_count_today: int = 0
    corrections_processed_today: int = 0
    idle_state: IdleState = IdleState.ACTIVE
    consecutive_errors: int = 0
    
    def update_from_observation(self, observation_type: str, intensity: float = 0.5):
        if observation_type == "correction_received":
            self.arousal = min(1.0, self.arousal + intensity * 0.1)
            self.energy = max(0.0, self.energy - intensity * 0.05)
            self.valence = EmotionalValence.NEUTRAL
        elif observation_type == "successful_task":
            self.energy = min(1.0, self.energy + intensity * 0.1)
            self.arousal = max(0.0, self.arousal - intensity * 0.05)
        elif observation_type == "error":
            self.energy = max(0.0, self.energy - intensity * 0.2)
            self.arousal = min(1.0, self.arousal + intensity * 0.2)
            self.consecutive_errors += 1
        elif observation_type == "user_positive":
            self.valence = EmotionalValence.POSITIVE
            self.energy = min(1.0, self.energy + intensity * 0.1)
        elif observation_type == "user_negative":
            self.valence = EmotionalValence.NEGATIVE
            self.energy = max(0.0, self.energy - intensity * 0.1)
    
    def check_sleeping(self, tz: str = "Europe/Berlin",
                        sleep_start: int = DEFAULT_SLEEP_START,
                        sleep_end: int = DEFAULT_SLEEP_END) -> bool:
        try:
            berlin = ZoneInfo(tz)
            now = datetime.now(berlin)
            current_hour = now.hour
            if current_hour >= sleep_start or current_hour < sleep_end:
                return True
            return False
        except Exception:
            return False
    
    def compute_idle_state(self, idle_thresholds: tuple = (300, 1800)) -> IdleState:
        now = datetime.now()
        seconds_idle = (now - self.last_user_activity).total_seconds()
        short_threshold, long_threshold = idle_thresholds
        if self.check_sleeping():
            return IdleState.SLEEPING
        elif seconds_idle < short_threshold:
            return IdleState.ACTIVE
        elif seconds_idle < long_threshold:
            return IdleState.IDLE_SHORT
        else:
            return IdleState.IDLE_LONG
    
    def reset_daily_counters(self):
        self.dream_count_today = 0
        self.nudge_count_today = 0
        self.corrections_processed_today = 0
    
    def to_dict(self) -> dict:
        return {
            "valence": self.valence.value,
            "energy": self.energy,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "idle_state": self.idle_state.value,
            "last_user_activity": self.last_user_activity.isoformat(),
            "dream_count_today": self.dream_count_today,
            "nudge_count_today": self.nudge_count_today,
            "corrections_processed_today": self.corrections_processed_today,
            "consecutive_errors": self.consecutive_errors,
        }

@dataclass  
class CircadianConfig:
    timezone: str = "Europe/Berlin"
    sleep_hours_start: int = 23
    sleep_hours_end: int = 7
    idle_thresholds: tuple = field(default_factory=lambda: (300, 1800))
    dream_intensity: Dict[IdleState, int] = field(default_factory=lambda: {
        IdleState.ACTIVE: 0,
        IdleState.IDLE_SHORT: 1,
        IdleState.IDLE_LONG: 3,
        IdleState.SLEEPING: 10,
        IdleState.FOCUS: 0,
    })
    correction_batch_interval: int = 60
    idle_check_interval: int = 30
    dream_cycle_interval: int = 300
    heartbeat_interval: int = 60
