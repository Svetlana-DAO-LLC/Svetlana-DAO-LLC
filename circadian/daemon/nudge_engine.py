"""Nudge engine — surfaces insights to main agent"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

class NudgePriority(Enum):
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    CRITICAL = 0.95

class NudgeCategory(Enum):
    MEMORY_CONSOLIDATION = "memory_consolidation"
    DREAM_INSIGHT = "dream_insight"
    CORRECTION_APPLIED = "correction_applied"
    PREFERENCE_LEARNED = "preference_learned"
    PATTERN_DETECTED = "pattern_detected"
    ENTITY_UPDATE = "entity_update"

@dataclass
class Nudge:
    content: str
    category: NudgeCategory
    priority: NudgePriority = NudgePriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""
    related_entities: List[str] = field(default_factory=list)
    confidence: float = 0.5
    shown: bool = False
    nudge_id: str = ""
    
    def __post_init__(self):
        if not self.nudge_id:
            import uuid
            self.nudge_id = str(uuid.uuid4())[:8]

class NudgeEngine:
    def __init__(self, max_queue_size: int = 50):
        self.max_queue_size = max_queue_size
        self._queue: deque[Nudge] = deque(maxlen=max_queue_size)
        self._high_priority_pending: List[Nudge] = []
        self._lock = asyncio.Lock()
    
    async def enqueue(self, nudge: Nudge, priority_override: Optional[float] = None) -> None:
        if priority_override is not None:
            nudge.priority = NudgePriority(priority_override)
        async with self._lock:
            if nudge.priority == NudgePriority.CRITICAL:
                self._high_priority_pending.append(nudge)
            else:
                self._queue.append(nudge)
    
    async def enqueue_dream_insight(self, insight: str, entities: List[str] = None, confidence: float = 0.6) -> None:
        nudge = Nudge(
            content=insight, category=NudgeCategory.DREAM_INSIGHT,
            priority=NudgePriority.MEDIUM, source="dream_cycle",
            related_entities=entities or [], confidence=confidence,
        )
        await self.enqueue(nudge)
    
    async def enqueue_correction_notification(self, wrong: str, correct: str) -> None:
        nudge = Nudge(
            content=f"Updated SOUL.md: {wrong} -> {correct}",
            category=NudgeCategory.CORRECTION_APPLIED,
            priority=NudgePriority.HIGH, source="correction_handler",
            confidence=0.95,
        )
        await self.enqueue(nudge)
    
    async def get_next(self) -> Optional[Nudge]:
        async with self._lock:
            if self._high_priority_pending:
                nudge = self._high_priority_pending.pop(0)
                nudge.shown = True
                return nudge
            if self._queue:
                nudge = self._queue.popleft()
                nudge.shown = True
                return nudge
        return None
    
    async def peek(self, count: int = 5) -> List[Nudge]:
        async with self._lock:
            return list(self._high_priority_pending) + list(self._queue)
    
    async def get_pending_count(self) -> int:
        async with self._lock:
            return len(self._high_priority_pending) + len(self._queue)
