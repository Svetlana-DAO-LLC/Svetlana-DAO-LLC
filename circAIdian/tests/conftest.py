"""Pytest fixtures for CircAIdian tests."""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("CIRCADIAN_API_TOKEN", "test-token")


@pytest.fixture
def temp_dirs():
    """Provide temp directories for soul, db, logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        soul = tmp / "SOUL.md"
        db = tmp / "corrections.db"
        dream_log = tmp / "dreams.log"
        nudge_log = tmp / "nudges.log"
        soul.write_text("# SOUL\n\n## AI Models\n- I use Claude\n## Preferences\n- User prefers concise responses\n")
        yield {"soul": soul, "db": db, "dream_log": dream_log, "nudge_log": nudge_log, "tmp": tmp}


@pytest.fixture
def correction_handler(temp_dirs):
    from daemon import CorrectionHandler
    return CorrectionHandler(
        soul_path=str(temp_dirs["soul"]),
        db_path=str(temp_dirs["db"]),  # Real temp file, not /dev/null
        batch_interval=1,
    )


@pytest.fixture
def nudge_engine():
    from daemon import NudgeEngine
    return NudgeEngine()


@pytest.fixture
def emotional_state():
    from daemon import EmotionalState
    return EmotionalState()


@pytest.fixture
def context_manager():
    from daemon import ActiveContextManager
    return ActiveContextManager(max_context_tokens=4000)


@pytest.fixture
def circadian_api(correction_handler, nudge_engine, context_manager, emotional_state):
    from daemon import CircadianAPI
    api = CircadianAPI(
        correction_handler=correction_handler,
        nudge_engine=nudge_engine,
        context_manager=context_manager,
        emotional_state=emotional_state,
        host="127.0.0.1",
        port=0,  # Random free port
    )
    return api


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each async test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
