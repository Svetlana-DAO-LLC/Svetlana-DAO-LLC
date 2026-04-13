"""CircAIdian — Svetlana Jr Subconscious + Dreaming + Active Learning Daemon"""
__version__ = "0.1.0"

from .correction_handler import CorrectionHandler, Correction
from .state import EmotionalState, CircadianConfig, IdleState
from .nudge_engine import NudgeEngine, Nudge, NudgePriority, NudgeCategory
from .active_context import ActiveContextManager, ContextChunk, ContextTier
from .api import CircadianAPI
