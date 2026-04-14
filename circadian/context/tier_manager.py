"""Active Context Manager — Tiered context retention with importance scoring"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
import re
import uuid

class ContextTier(Enum):
    PINNED = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class ContextChunk:
    content: str
    tier: ContextTier = ContextTier.MEDIUM
    importance_score: float = 0.5
    entity_tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    pinned: bool = False
    is_correction: bool = False
    is_preference: bool = False
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())[:8]

    def access(self):
        self.last_accessed = datetime.now()

    def estimate_tokens(self) -> int:
        return len(self.content) // 4

class ActiveContextManager:
    # Tuning constants
    ONE_HOUR = 3600
    TEN_MINUTES = 600
    ENTITY_RECENCY_WEIGHT = 0.2
    ACCESS_BOOST_MAX = 0.05

    _CORRECTION_PATTERNS = (
        "not claude", "not chatgpt", "not anthropic", "don't have claude",
        "no, it's", "actually it's", "wait, it's", "correction:",
    )

    def __init__(self, max_context_tokens: int = 16000):
        self.max_tokens = max_context_tokens
        self.tiers: Dict[ContextTier, List[ContextChunk]] = {t: [] for t in ContextTier}
        self.entity_recency: Dict[str, datetime] = {}
        self.current_task: str = ""

    @property
    def pinned_chunks(self) -> List[ContextChunk]:
        return self.tiers[ContextTier.PINNED]

    @property
    def total_chunks(self) -> int:
        return sum(len(t) for t in self.tiers.values())

    @property
    def estimated_tokens(self) -> int:
        return sum(c.estimate_tokens() for t in self.tiers.values() for c in t)

    def add(self, chunk: ContextChunk) -> None:
        content_lower = chunk.content.lower()
        if any(p in content_lower for p in self._CORRECTION_PATTERNS):
            chunk.is_correction = True
            chunk.pinned = True
            chunk.tier = ContextTier.PINNED
        elif any(kw in content_lower for kw in ["prefer", "always", "never", "don't", "user"]):
            chunk.is_preference = True
            chunk.pinned = True
            chunk.tier = ContextTier.PINNED
        for entity in self._extract_entities(chunk.content):
            chunk.entity_tags.append(entity)
            self.entity_recency[entity] = datetime.now()
        self.tiers[chunk.tier].append(chunk)
        self._rebalance()

    def add_message_pair(self, user_msg: str, agent_response: str) -> None:
        chunk = ContextChunk(
            content=f"User: {user_msg}\nAgent: {agent_response}",
            tier=ContextTier.MEDIUM,
        )
        for text in [user_msg, agent_response]:
            for entity in self._extract_entities(text):
                chunk.entity_tags.append(entity)
                self.entity_recency[entity] = datetime.now()
        self.add(chunk)

    def add_correction(self, wrong: str, correct: str, source: str = "user") -> None:
        chunk = ContextChunk(
            content=f"[CORRECTION from {source}]: Was WRONG: {wrong}. CORRECT: {correct}",
            tier=ContextTier.PINNED, importance_score=0.99, pinned=True, is_correction=True,
        )
        self.tiers[ContextTier.PINNED].append(chunk)

    def add_user_preference(self, preference: str) -> None:
        chunk = ContextChunk(
            content=f"[USER PREFERENCE]: {preference}",
            tier=ContextTier.PINNED, importance_score=0.95, pinned=True, is_preference=True,
        )
        self.tiers[ContextTier.PINNED].append(chunk)

    def get_all_chunks(self) -> List[ContextChunk]:
        """Return all chunks across all tiers as a list."""
        all_chunks = []
        for tier_chunks in self.tiers.values():
            all_chunks.extend(tier_chunks)
        return all_chunks

    def get_context_for_prompt(self) -> str:
        chunks = []
        tokens_used = 0
        max_pinned_tokens = int(self.max_tokens * 0.25)
        for chunk in self.tiers[ContextTier.PINNED]:
            if tokens_used + chunk.estimate_tokens() < max_pinned_tokens:
                chunk.access()
                chunks.append(chunk)
                tokens_used += chunk.estimate_tokens()
        max_high_tokens = int(self.max_tokens * 0.50)
        for chunk in self._ranked(self.tiers[ContextTier.HIGH]):
            if tokens_used + chunk.estimate_tokens() < max_high_tokens:
                chunk.access()
                chunks.append(chunk)
                tokens_used += chunk.estimate_tokens()
        max_medium_tokens = int(self.max_tokens * 0.95)
        for chunk in self._ranked(self.tiers[ContextTier.MEDIUM]):
            if tokens_used + chunk.estimate_tokens() < max_medium_tokens:
                chunk.access()
                chunks.append(chunk)
                tokens_used += chunk.estimate_tokens()
        return self._assemble_context(chunks)

    def _ranked(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        now = datetime.now()
        def composite_score(chunk: ContextChunk) -> float:
            score = chunk.importance_score
            if chunk.entity_tags:
                entity_recency_boost = 0.0
                for entity in chunk.entity_tags:
                    if entity in self.entity_recency:
                        seconds_ago = (now - self.entity_recency[entity]).total_seconds()
                        recency_score = max(0, 1 - seconds_ago / self.ONE_HOUR)
                        entity_recency_boost = max(entity_recency_boost, recency_score * self.ENTITY_RECENCY_WEIGHT)
                score += entity_recency_boost
            seconds_since_access = (now - chunk.last_accessed).total_seconds()
            access_boost = max(0, self.ACCESS_BOOST_MAX * (1 - seconds_since_access / self.TEN_MINUTES))
            score += access_boost
            return min(score, 1.0)
        return sorted(chunks, key=composite_score, reverse=True)

    def _rebalance(self) -> None:
        if self.estimated_tokens <= self.max_tokens:
            return
        medium = self.tiers[ContextTier.MEDIUM]
        medium.sort(key=lambda c: c.importance_score)
        while self.estimated_tokens > self.max_tokens and medium:
            demoted = medium.pop(0)
            demoted.tier = ContextTier.LOW
            self.tiers[ContextTier.LOW].append(demoted)

    def _assemble_context(self, chunks: List[ContextChunk]) -> str:
        if not chunks:
            return ""
        parts = ["[CONTEXT — importance-ordered]"]
        for chunk in chunks:
            tier_label = f"[{chunk.tier.name}]"
            if chunk.is_correction:
                tier_label = "[CORRECTION]"
            elif chunk.is_preference:
                tier_label = "[PREFERENCE]"
            parts.append(f"{tier_label} {chunk.content}")
        return "\n".join(parts)

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        entities.extend(capitalized[:5])
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend([q for q in quoted if len(q) > 2][:3])
        return list(set(entities))[:10]

    def set_current_task(self, task: str) -> None:
        self.current_task = task

    def clear_low_tier(self) -> int:
        count = len(self.tiers[ContextTier.LOW])
        self.tiers[ContextTier.LOW] = []
        return count

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "estimated_tokens": self.estimated_tokens,
            "max_tokens": self.max_tokens,
            "tiers": {t.name: len(self.tiers[t]) for t in ContextTier},
            "entity_count": len(self.entity_recency),
        }
