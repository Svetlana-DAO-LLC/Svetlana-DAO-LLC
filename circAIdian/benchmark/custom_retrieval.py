"""
Custom CircAIdian retrieval powered by MiniMax M2.7.

Three-stage retrieval:
  1. BM25        — fast keyword baseline
  2. HYDE        — M2.7 generates hypothetical answer, concat with query → BM25
  3. Subconscious + Rerank — M2.7 processes turns as "dreams", then reranks BM25 results

All LLM calls go through `mmx text chat --model MiniMax-M2.7`.

Usage:
    python -m benchmark.harness --mode custom
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from benchmark.bm25 import BM25, bm25_retrieve


# ---------------------------------------------------------------------------
# MiniMax M2.7 via mmx CLI
# ---------------------------------------------------------------------------

MMX_CMD = "mmx"


def _call_minimax(prompt: str, max_tokens: int = 200, timeout: int = 30) -> Optional[str]:
    """
    Call MiniMax M2.7 via mmx text chat.

    Returns the response text, or None on failure.
    """
    try:
        result = subprocess.run(
            [
                MMX_CMD, "text", "chat",
                "--message", prompt,
                "--model", "MiniMax-M2.7",
                "--max-tokens", str(max_tokens),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        for item in data.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "").strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# HYDE — Hypothetical Document Embeddings
# ---------------------------------------------------------------------------

HYDE_SYSTEM = (
    "You are a hypothetical answer generator. Given a question about "
    "a person's preferences, history, habits, opinions, or other personal facts, "
    "generate a realistic hypothetical answer as if that person had answered the "
    "question themselves. Keep answers brief (1-3 sentences) and plausible.\n"
    "Do NOT say 'I don\\'t know' or 'undisclosed' or 'I\\'m an AI' — "
    "generate a plausible, specific answer grounded in what the question asks."
)


def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for a query using MiniMax M2.7."""
    prompt = f"{HYDE_SYSTEM}\n\nQuestion: {query}\nHypothetical Answer:"
    result = _call_minimax(prompt, max_tokens=200)
    return result or ""


# ---------------------------------------------------------------------------
# Subconscious / Dream processing
# ---------------------------------------------------------------------------

SUBconscious_SYSTEM = (
    "You are CircAIdian's subconscious — a latent reasoning engine that processes "
    "conversation history to surface latent patterns, associations, and implied facts. "
    "Given a conversation, generate a brief 'dream narrative' — a stream-of-consciousness "
    "summary of implied preferences, habits, emotional patterns, and unstated facts.\n"
    "Focus on: specific names, numbers, dates, preferences, opinions implied by tone.\n"
    "Be concise. Output 3-5 sentences of free association."
)


def generate_subconscious_dream(turns: List[Tuple[str, str]], current_query: str) -> str:
    """
    Generate a 'subconscious dream' from conversation turns using MiniMax M2.7.

    This processes the full conversation context to surface latent patterns,
    then the result is used as a reranking signal.
    """
    # Format turns for the model
    turns_text = "\n".join(
        f"User: {u}\nAgent: {a}" for u, a in turns
    )
    prompt = (
        f"{SUBconscious_SYSTEM}\n\n"
        f"Conversation:\n{turns_text}\n\n"
        f"Current question being asked: {current_query}\n\n"
        f"Subconscious dream (what patterns does this conversation suggest?):"
    )
    result = _call_minimax(prompt, max_tokens=300)
    return result or ""


# ---------------------------------------------------------------------------
# Reranker — uses MiniMax M2.7 to score and reorder BM25 results
# ---------------------------------------------------------------------------

RERANK_SYSTEM = (
    "You are a relevance reranker. Given a question and a list of retrieved text chunks, "
    "score each chunk 0-1 for how well it answers the question. "
    "Output ONLY a valid JSON array like [0.9, 0.3, 0.1, 0.8, 0.2] — one score per chunk "
    "in the same order given. Higher = more relevant. "
    "If a chunk is completely irrelevant or hallucinated, score it 0.0."
)


def rerank_chunks(
    query: str,
    chunks: List[Tuple[str, str]],
    scores: List[float],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Rerank (chunk_id, content) pairs using MiniMax M2.7.

    Takes BM25 initial scores and reorders using semantic relevance.
    Returns reranked list of (chunk_id, new_score) pairs.
    """
    if not chunks:
        return []

    # Format chunks for the model
    chunks_text = "\n".join(
        f"[{i}] {cid}: {content[:200]}"
        for i, (cid, content) in enumerate(chunks)
    )
    prompt = (
        f"{RERANK_SYSTEM}\n\n"
        f"Question: {query}\n\n"
        f"Retrieved chunks:\n{chunks_text}\n\n"
        f"Scores (BM25 initial): {scores}\n\n"
        f"JSON array of relevance scores:"
    )

    raw = _call_minimax(prompt, max_tokens=100)
    if not raw:
        return list(zip([c[0] for c in chunks], scores))

    # Parse JSON array from response
    try:
        # Try to extract array from response
        match = re.search(r'\[\s*([\d.,\s]+)\s*\]', raw)
        if match:
            rerank_scores = [float(x.strip()) for x in match.group(1).split(",")]
        else:
            data = json.loads(raw)
            rerank_scores = list(data) if isinstance(data, list) else []
    except Exception:
        rerank_scores = []

    if len(rerank_scores) != len(chunks):
        return list(zip([c[0] for c in chunks], scores))

    # Fuse BM25 scores with rerank scores (weighted geometric mean)
    fused = []
    for i, (cid, content) in enumerate(chunks):
        bm25_s = scores[i] if i < len(scores) else 0.0
        rerank_s = rerank_scores[i] if i < len(rerank_scores) else 0.0
        # Geometric mean — both must be high to rank well
        fused_s = (bm25_s * rerank_s) ** 0.5 if bm25_s > 0 and rerank_s > 0 else 0.0
        fused.append((cid, fused_s))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# ---------------------------------------------------------------------------

class CircAIdianRetrieval:
    """
    Three-stage retrieval:
      1. BM25 baseline
      2. HYDE expansion (M2.7 generates hypothetical answer → concat → BM25)
      3. Subconscious + Rerank (M2.7 dream + M2.7 rerank)

    Drop-in replacement for BM25 / HYDEBM25Retrieval in the harness.
    """

    def __init__(self):
        self._ranker: Optional[BM25] = None
        self._chunks: List[Tuple[str, str]] = []
        self._chunk_map: dict = {}
        self._hyde_cache: dict = {}
        self._turns: List[Tuple[str, str]] = []

    def index(self, chunks: List[Tuple[str, str]], turns: List[Tuple[str, str]] = None):
        """Index chunks for retrieval. Must be called before query()."""
        if not chunks:
            return
        self._chunks = chunks
        self._turns = turns or []
        self._chunk_map = {cid: content for cid, content in chunks}
        doc_ids = [cid for cid, _ in chunks]
        doc_texts = [content for _, content in chunks]
        self._ranker = BM25(k1=1.5, b=0.75)
        self._ranker.index(doc_ids, doc_texts)

    async def query_async(self, query: str, top_k: int = 5) -> str:
        """
        Full three-stage retrieval:
          1. BM25 initial retrieval
          2. HYDE expansion (M2.7)
          3. Subconscious dream (M2.7) + Rerank (M2.7)
        """
        if not self._ranker:
            return ""

        # ---- Stage 1: BM25 initial retrieval ----
        bm25_results = self._ranker.retrieve(query, top_k=top_k * 2)
        initial_chunks = [(cid, self._chunk_map.get(cid, "")) for cid, _ in bm25_results]
        initial_scores = [s for _, s in bm25_results]

        # ---- Stage 2: HYDE expansion — generate hypothetical answer ----
        hyde_doc = ""
        if query not in self._hyde_cache:
            loop = asyncio.get_event_loop()
            self._hyde_cache[query] = await loop.run_in_executor(
                None, generate_hypothetical_answer, query
            )
        hyde_doc = self._hyde_cache[query]

        # Re-run BM25 with HYDE expansion
        search_query = f"{query} {hyde_doc}".strip() if hyde_doc else query
        hyde_results = self._ranker.retrieve(search_query, top_k=top_k * 2)
        hyde_chunks = [(cid, self._chunk_map.get(cid, "")) for cid, _ in hyde_results]
        hyde_scores = [s for _, s in hyde_results]

        # ---- Stage 3: Subconscious dream + Rerank ----
        if self._turns:
            loop = asyncio.get_event_loop()
            dream = await loop.run_in_executor(
                None, generate_subconscious_dream, self._turns, query
            )
            if dream:
                # Re-run BM25 with dream context
                dream_query = f"{query} {dream}".strip()
                dream_results = self._ranker.retrieve(dream_query, top_k=top_k * 2)
                dream_chunks = [(cid, self._chunk_map.get(cid, "")) for cid, _ in dream_results]
                dream_scores = [s for _, s in dream_results]

                # Combine: fuse hyde + dream BM25 scores
                combined = {}
                for cid, s in zip([c[0] for c in hyde_chunks], hyde_scores):
                    combined[cid] = combined.get(cid, 0.0) + s * 0.6
                for cid, s in zip([c[0] for c in dream_chunks], dream_scores):
                    combined[cid] = combined.get(cid, 0.0) + s * 0.4

                top_cids = sorted(combined, key=combined.get, reverse=True)[:top_k * 2]
                fused_chunks = [(cid, self._chunk_map.get(cid, "")) for cid in top_cids]
                fused_scores = [combined[cid] for cid in top_cids]
            else:
                fused_chunks = hyde_chunks
                fused_scores = hyde_scores
        else:
            fused_chunks = hyde_chunks
            fused_scores = hyde_scores

        # ---- Rerank with MiniMax M2.7 ----
        reranked = await asyncio.get_event_loop().run_in_executor(
            None, rerank_chunks, query, fused_chunks, fused_scores, top_k
        )

        # Map back to content
        result_cids = [cid for cid, _ in reranked if cid in self._chunk_map]
        retrieved = " ".join(self._chunk_map.get(cid, "") for cid in result_cids)
        return retrieved if retrieved else bm25_retrieve(query, self._chunks, top_k)

    def query(self, query: str, top_k: int = 5) -> str:
        """Synchronous wrapper."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.query_async(query, top_k))


# ---------------------------------------------------------------------------
# Simpler two-stage (HYDE + Rerank, no subconscious) for faster benchmarking
# ---------------------------------------------------------------------------

class CircAIdianLightRetrieval:
    """
    Two-stage retrieval (faster):
      1. BM25 baseline
      2. HYDE expansion (M2.7) → re-BM25 → M2.7 rerank

    Simpler than full CircAIdianRetrieval — skips subconscious processing.
    """

    def __init__(self):
        self._ranker: Optional[BM25] = None
        self._chunks: List[Tuple[str, str]] = []
        self._chunk_map: dict = {}
        self._hyde_cache: dict = {}

    def index(self, chunks: List[Tuple[str, str]]):
        if not chunks:
            return
        self._chunks = chunks
        self._chunk_map = {cid: content for cid, content in chunks}
        doc_ids = [cid for cid, _ in chunks]
        doc_texts = [content for _, content in chunks]
        self._ranker = BM25(k1=1.5, b=0.75)
        self._ranker.index(doc_ids, doc_texts)

    async def query_async(self, query: str, top_k: int = 5) -> str:
        if not self._ranker:
            return ""

        # ---- Stage 1: BM25 baseline ----
        bm25_results = self._ranker.retrieve(query, top_k=top_k * 4)
        bm25_scores = [s for _, s in bm25_results]
        bm25_cids = [cid for cid, _ in bm25_results]

        # ---- Stage 2: HYDE expansion (M2.7) ----
        if query not in self._hyde_cache:
            loop = asyncio.get_event_loop()
            self._hyde_cache[query] = await loop.run_in_executor(
                None, generate_hypothetical_answer, query
            )
        hyde_doc = self._hyde_cache[query]

        search_query = f"{query} {hyde_doc}".strip() if hyde_doc else query
        hyde_results = self._ranker.retrieve(search_query, top_k=top_k * 4)
        hyde_scores = [s for _, s in hyde_results]
        hyde_cids = [cid for cid, _ in hyde_results]

        # ---- Fuse: BM25*0.4 + HYDE*0.6 ----
        # Balance keyword precision with semantic expansion
        fused = {}
        for cid, s in zip(bm25_cids, bm25_scores):
            fused[cid] = fused.get(cid, 0.0) + s * 0.4
        for cid, s in zip(hyde_cids, hyde_scores):
            fused[cid] = fused.get(cid, 0.0) + s * 0.6

        top_cids = sorted(fused, key=fused.get, reverse=True)[:top_k * 3]
        candidates = [(cid, self._chunk_map.get(cid, "")) for cid in top_cids]
        candidate_scores = [fused[cid] for cid in top_cids]

        # ---- Stage 3: M2.7 rerank ----
        reranked = await asyncio.get_event_loop().run_in_executor(
            None, rerank_chunks, query, candidates, candidate_scores, top_k
        )

        result_cids = [cid for cid, _ in reranked if cid in self._chunk_map]
        retrieved = " ".join(self._chunk_map.get(cid, "") for cid in result_cids)
        return retrieved if retrieved else bm25_retrieve(query, self._chunks, top_k)

    def query(self, query: str, top_k: int = 5) -> str:
        """Synchronous wrapper."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.query_async(query, top_k))
