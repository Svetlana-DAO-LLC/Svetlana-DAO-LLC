"""CircAIdian daemon package"""
__version__ = "0.1.0"

from .correction_handler import CorrectionHandler, Correction
from .state import EmotionalState, IdleState, EmotionalValence
from .nudge_engine import NudgeEngine, Nudge, NudgePriority, NudgeCategory
from .api import CircadianAPI

# CircadianConfig lives in config/__init__.py — re-export for convenience.
from config import CircadianConfig  # noqa: F401

# ActiveContextManager lives in context.tier_manager — re-export for convenience.
from context.tier_manager import ActiveContextManager, ContextChunk, ContextTier  # noqa: F401
