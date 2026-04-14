"""
HYDE (Hypothetical Document Embeddings) + BM25 retrieval for CircAIdian.

Uses MiniMax M2.7 via `mmx text chat` for fast HYDE query expansion.
Falls back to pure BM25 if mmx is unavailable.

HYDE (Rule et al., 2022): generate a hypothetical answer document from
the query, then retrieve chunks that match the hypothetical document.
This gives semantic expansion without needing vector embeddings.
"""
import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

from benchmark.bm25 import BM25, bm25_retrieve


# mmx CLI for MiniMax M2.7 inference
MMX_CMD = "mmx"


def _call_minimax(prompt: str, timeout: int = 30) -> Optional[str]:
    """
    Call MiniMax M2.7 via mmx text chat to generate a hypothetical answer.

    Returns the answer text, or None on failure.
    """
    try:
        result = subprocess.run(
            [
                MMX_CMD, "text", "chat",
                "--message", prompt,
                "--model", "MiniMax-M2.7",
                "--max-tokens", "200",
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        # Extract text content from MiniMax response format
        # [{"type": "thinking", ...}, {"type": "text", "text": "..."}]
        for item in data.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "").strip()
    except Exception:
        pass
    return None


HYDE_SYSTEM_PROMPT = """You are a hypothetical answer generator. Given a question about
a person's preferences, history, habits, opinions, or other personal facts,
generate a realistic hypothetical answer as if that person had answered the
question themselves. Keep answers brief (1-3 sentences) and plausible.
If the question asks about information that could be private or sensitive,
give a natural short answer that sounds like what that person might say.
Do NOT say "I don't know" or "undisclosed" — generate a plausible answer."""


def generate_hypothetical_answer(query: str) -> str:
    """
    Generate a hypothetical answer document for a query using MiniMax M2.7.

    Returns the hypothetical answer text, or empty string on failure.
    """
    prompt = f"{HYDE_SYSTEM_PROMPT}\n\nQuestion: {query}\nHypothetical Answer:"
    result = _call_minimax(prompt)
    if result:
        return result
    return ""


async def generate_hypothetical_answer_async(query: str) -> str:
    """Async wrapper — runs mmx in executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_hypothetical_answer, query)


def hyde_retrieve(
    query: str,
    chunks: List[Tuple[str, str]],
    top_k: int = 5,
    use_hyde: bool = True,
) -> str:
    """
    HYDE + BM25 retrieval (synchronous).

    Strategy:
    1. If use_hyde=True: generate hypothetical answer with MiniMax M2.7
    2. Run BM25 with both query AND hypothetical answer against chunks
    3. Fall back to pure BM25 if HYDE generation fails

    Args:
        query: The user's question
        chunks: List of (chunk_id, content) tuples
        top_k: Number of top results to return
        use_hyde: Whether to attempt HYDE expansion

    Returns:
        Concatenated text of retrieved chunks
    """
    if not chunks:
        return ""

    hyde_doc = ""
    if use_hyde:
        hyde_doc = generate_hypothetical_answer(query)

    # Run BM25 with both query and hypothetical doc
    search_query = f"{query} {hyde_doc}".strip() if hyde_doc else query
    return bm25_retrieve(search_query, chunks, top_k)


async def hyde_retrieve_async(
    query: str,
    chunks: List[Tuple[str, str]],
    top_k: int = 5,
) -> str:
    """
    Async HYDE + BM25 retrieval.

    Generates hypothetical answer concurrently while preparing BM25 index.
    """
    if not chunks:
        return ""

    # Generate HYDE document in background
    loop = asyncio.get_event_loop()
    hyde_future = loop.run_in_executor(None, generate_hypothetical_answer, query)

    # Build BM25 index synchronously while waiting
    doc_ids = [cid for cid, _ in chunks]
    doc_texts = [content for _, content in chunks]
    ranker = BM25(k1=1.5, b=0.75)
    ranker.index(doc_ids, doc_texts)

    # Wait for HYDE
    hyde_doc = await hyde_future

    search_query = f"{query} {hyde_doc}".strip() if hyde_doc else query
    results = ranker.retrieve(search_query, top_k=top_k)

    chunk_map = {cid: content for cid, content in chunks}
    retrieved = " ".join(chunk_map.get(cid, "") for cid, _ in results if cid in chunk_map)
    return retrieved if retrieved else bm25_retrieve(query, chunks, top_k)


class HYDEBM25Retrieval:
    """
    HYDE + BM25 retriever with async HYDE generation.

    Drop-in replacement for bm25_retrieve in the benchmark harness.
    """

    def __init__(self):
        self._ranker: Optional[BM25] = None
        self._chunks: List[Tuple[str, str]] = []
        self._chunk_map: dict = {}
        self._hyde_cache: dict = {}

    def index(self, chunks: List[Tuple[str, str]]):
        """Index chunks for retrieval. Must be called before query()."""
        if not chunks:
            return
        self._chunks = chunks
        self._chunk_map = {cid: content for cid, content in chunks}
        doc_ids = [cid for cid, _ in chunks]
        doc_texts = [content for _, content in chunks]
        self._ranker = BM25(k1=1.5, b=0.75)
        self._ranker.index(doc_ids, doc_texts)

    async def query_async(self, query: str, top_k: int = 5) -> str:
        """Retrieve using HYDE + BM25 (async HYDE generation)."""
        if not self._ranker:
            return ""

        # Check cache
        if query not in self._hyde_cache:
            self._hyde_cache[query] = await asyncio.get_event_loop().run_in_executor(
                None, generate_hypothetical_answer, query
            )
        hyde_doc = self._hyde_cache[query]

        search_query = f"{query} {hyde_doc}".strip() if hyde_doc else query
        results = self._ranker.retrieve(search_query, top_k=top_k)

        retrieved = " ".join(
            self._chunk_map.get(cid, "") for cid, _ in results if cid in self._chunk_map
        )
        return retrieved if retrieved else bm25_retrieve(query, self._chunks, top_k)

    def query(self, query: str, top_k: int = 5) -> str:
        """Synchronous wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.query_async(query, top_k)
        )
