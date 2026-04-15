"""Tests for EmotionalState."""
import pytest
from datetime import datetime


class TestEmotionalState:
    def test_default_state(self, emotional_state):
        assert emotional_state.energy == 0.5
        assert emotional_state.arousal == 0.3
        assert emotional_state.valence.value == "neutral"
        assert emotional_state.idle_state.value == "active"

    def test_correction_received(self, emotional_state):
        emotional_state.update_from_observation("correction_received", intensity=0.5)
        assert emotional_state.arousal > 0.3
        assert emotional_state.energy < 0.5
        assert emotional_state.valence.value == "neutral"

    def test_successful_task(self, emotional_state):
        emotional_state.update_from_observation("successful_task", intensity=0.5)
        assert emotional_state.energy > 0.5
        assert emotional_state.arousal < 0.3

    def test_error(self, emotional_state):
        emotional_state.update_from_observation("error", intensity=0.5)
        assert emotional_state.energy < 0.5
        assert emotional_state.arousal > 0.3
        assert emotional_state.consecutive_errors == 1

    def test_user_positive(self, emotional_state):
        emotional_state.update_from_observation("user_positive", intensity=0.5)
        assert emotional_state.valence.value == "positive"

    def test_user_negative(self, emotional_state):
        emotional_state.update_from_observation("user_negative", intensity=0.5)
        assert emotional_state.valence.value == "negative"

    def test_compute_idle_state_short(self, emotional_state):
        # User was active recently
        emotional_state.last_user_activity = datetime.now()
        state = emotional_state.compute_idle_state((300, 1800))
        assert state.value == "active"

    def test_compute_idle_state_long(self, emotional_state):
        # User idle > 1800s
        emotional_state.last_user_activity = datetime.fromtimestamp(0)
        state = emotional_state.compute_idle_state((300, 1800))
        assert state.value == "idle_long"

    def test_to_dict(self, emotional_state):
        d = emotional_state.to_dict()
        assert "valence" in d
        assert "energy" in d
        assert "idle_state" in d
        assert d["valence"] == "neutral"

    def test_reset_daily_counters(self, emotional_state):
        emotional_state.dream_count_today = 5
        emotional_state.nudge_count_today = 3
        emotional_state.corrections_processed_today = 2
        emotional_state.reset_daily_counters()
        assert emotional_state.dream_count_today == 0
        assert emotional_state.nudge_count_today == 0
        assert emotional_state.corrections_processed_today == 0

    def test_check_sleeping_below_threshold(self, emotional_state):
        # Use non-sleep hours (e.g. noon in Berlin)
        result = emotional_state.check_sleeping(tz="Europe/Berlin", sleep_start=23, sleep_end=7)
        # Noon is not sleep time
        assert result is False

    def test_check_sleeping_invalid_tz(self, emotional_state):
        # Invalid timezone should return False, not raise
        result = emotional_state.check_sleeping(tz="Invalid/Timezone")
        assert result is False
