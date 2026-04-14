"""Tests for ActiveContextManager."""
import pytest
from datetime import datetime, timedelta


class TestActiveContextManager:
    def test_add_chunk_default_tier(self, context_manager):
        from daemon import ContextChunk, ContextTier
        chunk = ContextChunk(content="Hello world")
        context_manager.add(chunk)
        assert chunk.tier == ContextTier.MEDIUM

    def test_add_correction_goes_pinned(self, context_manager):
        from daemon import ContextChunk
        chunk = ContextChunk(content="Actually it's not Claude, it's OpenCode")
        context_manager.add(chunk)
        assert chunk.is_correction is True
        assert chunk.pinned is True

    def test_add_preference_goes_pinned(self, context_manager):
        from daemon import ContextChunk
        chunk = ContextChunk(content="User prefers short responses")
        context_manager.add(chunk)
        assert chunk.is_preference is True
        assert chunk.pinned is True

    def test_context_for_prompt_empty(self, context_manager):
        result = context_manager.get_context_for_prompt()
        assert result == ""

    def test_context_for_prompt_contains_chunks(self, context_manager):
        from daemon import ContextChunk
        context_manager.add(ContextChunk(content="First message"))
        context_manager.add(ContextChunk(content="Second message"))
        result = context_manager.get_context_for_prompt()
        assert "First message" in result or "Second message" in result

    def test_estimated_tokens(self, context_manager):
        from daemon import ContextChunk
        context_manager.add(ContextChunk(content="x" * 100))
        assert context_manager.estimated_tokens > 0

    def test_get_stats(self, context_manager):
        from daemon import ContextChunk
        context_manager.add(ContextChunk(content="test"))
        stats = context_manager.get_stats()
        assert "total_chunks" in stats
        assert "estimated_tokens" in stats
        assert stats["total_chunks"] == 1

    def test_clear_low_tier(self, context_manager):
        from daemon import ContextChunk, ContextTier
        chunk = ContextChunk(content="low priority content")
        context_manager.add(chunk)
        # Manually move to LOW tier
        context_manager.tiers[ContextTier.MEDIUM].remove(chunk)
        context_manager.tiers[ContextTier.LOW].append(chunk)
        count = context_manager.clear_low_tier()
        assert count == 1
        assert len(context_manager.tiers[ContextTier.LOW]) == 0

    def test_add_message_pair(self, context_manager):
        context_manager.add_message_pair("User said hello", "Agent responded")
        assert context_manager.total_chunks >= 1

    def test_add_correction_method(self, context_manager):
        context_manager.add_correction("wrong thing", "correct thing", "test")
        assert context_manager.total_chunks == 1

    def test_add_user_preference_method(self, context_manager):
        context_manager.add_user_preference("User likes JSON")
        assert context_manager.total_chunks == 1
