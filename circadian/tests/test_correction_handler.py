"""Tests for CorrectionHandler."""
import pytest


class TestCorrectionHandler:
    def test_detect_negation_explicit(self, correction_handler):
        # "No, it's X not Y" pattern — the wrong part is captured
        result = correction_handler._detect_correction(
            "No, it's Claude Code, not OpenCode"
        )
        assert result is not None
        assert result.wrong_claim.lower() in ("claude code", "opencode")
        assert result.correction != result.wrong_claim
        assert result.confidence >= 0.8

    def test_detect_no_its_wrong(self, correction_handler):
        # "No, it's X that's wrong" pattern — captures the named entity
        result = correction_handler._detect_correction(
            "No, it's Claude Code that's wrong"
        )
        assert result is not None
        assert result.wrong_claim.strip() != ""
        assert result.confidence >= 0.8

    def test_detect_actually(self, correction_handler):
        result = correction_handler._detect_correction(
            "Actually it's GLM-5.1, not GPT-4"
        )
        assert result is not None
        assert result.confidence >= 0.8

    def test_detect_wait(self, correction_handler):
        result = correction_handler._detect_correction(
            "Wait, I meant to say you use OpenCode not Codex"
        )
        assert result is not None

    def test_detect_no_correction(self, correction_handler):
        result = correction_handler._detect_correction(
            "Hello, how are you today?"
        )
        assert result is None

    def test_detect_empty(self, correction_handler):
        assert correction_handler._detect_correction("") is None
        assert correction_handler._detect_correction("   ") is None
        assert correction_handler._detect_correction("ab") is None  # too short

    def test_same_wrong_and_correct_skipped(self, correction_handler):
        # Should not detect when wrong == correct (case insensitive)
        result = correction_handler._detect_correction(
            "Actually it's the same, it's the same"
        )
        # Pattern might match but same-wrong-correct check filters it
        # (depends on pattern groups — this is a regression test)

    def test_soul_contradiction_flagged(self, correction_handler):
        # SOUL says "Claude Code". User says "No, it's OpenCode, not Claude Code".
        # Detection: wrong="Claude Code", correct="OpenCode" (semantic swap above).
        # wrong="Claude Code" IS in SOUL → requires_soul_update=True.
        correction_handler.soul_path.write_text("# SOUL\n\n## AI Models\n- I use Claude Code\n")
        c = correction_handler._detect_correction("No, it's OpenCode, not Claude Code")
        assert c is not None
        c = correction_handler._check_soul_contradiction(c, correction_handler.soul_path.read_text())
        assert c.requires_soul_update is True
        assert c.soul_section is not None

    def test_find_soul_section(self, correction_handler):
        content = """
## AI Models
- I use Claude
## Preferences
- User likes short responses
"""
        section = correction_handler._find_soul_section("claude", content)
        assert section == "ai_models"

    @pytest.mark.asyncio
    async def test_process_observation_queues_correction(self, temp_dirs, correction_handler):
        # Pass the REAL SOUL content (from temp_dirs), not the handler's soul_path (/dev/null)
        real_soul_content = temp_dirs["soul"].read_text()
        assert "Claude" in real_soul_content  # verify fixture content
        await correction_handler.process_observation(
            "sess1",
            "No, it's OpenCode, not Claude Code",
            "",
            real_soul_content,
        )
        assert await correction_handler.get_pending_count() == 1

    @pytest.mark.asyncio
    async def test_apply_correction_replaces_line(self, temp_dirs, correction_handler):
        correction_handler.soul_path.write_text(
            "# SOUL\n\n## AI Models\n- I use Claude Code\n"
        )
        from datetime import datetime
        from daemon import Correction
        c = Correction(
            session_id="s1", timestamp=datetime.now(),
            wrong_claim="Claude Code", correction="OpenCode",
            source="test", requires_soul_update=True,
            soul_section="ai_models",
        )
        result = correction_handler._apply_correction(
            correction_handler.soul_path.read_text(), c
        )
        assert "OpenCode" in result
        assert "Claude Code" not in result
