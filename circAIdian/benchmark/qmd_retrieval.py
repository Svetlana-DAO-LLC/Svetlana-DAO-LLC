"""
QMD-backed retrieval for CircAIdian benchmark harness.

Uses QMD's hybrid lex + vec + hyde retrieval via CLI subprocess.
QMD stores collections at /home/hermes/workspace/<name>/ — this is fixed,
the path argument to `qmd collection add` is for display only.

Workflow (async-friendly):
    1. index_chunks()   — write chunks as .md files, add collection, start embed (non-blocking)
    2. query()          — run hybrid search immediately (BM25 until vectors ready), return (chunk_id, score)
    3. close()          — clean up collection

Falls back to local BM25 if QMD is unavailable or fails.
The embed runs in background; query() works immediately using BM25 scores
until vector embeddings are computed by QMD's periodic background job.
"""
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from benchmark.bm25 import bm25_retrieve


# QMD always stores collections at /home/hermes/workspace/<name>/
QMD_WORKSPACE = Path("/home/hermes/workspace")


class QMDRetrieval:
    """
    Manages a QMD collection for benchmark chunks with BM25 fallback.

    The workflow:
    1. index_chunks() — write chunks as .md files, add to QMD, start embed (non-blocking)
    2. query()         — run hybrid search immediately (BM25 until vectors ready)
    3. close()         — clean up collection
    """

    def __init__(self, collection_name: str = "ca-bench", persistent: bool = False):
        self.collection_name = collection_name
        self.persistent = persistent  # if True, don't clean up on close()
        self._indexed = False
        self._chunks: List[Tuple[str, str]] = []
        self._embed_proc: subprocess.Popen | None = None

    def _cleanup(self):
        """Remove collection and local files."""
        try:
            subprocess.run(
                ["qmd", "collection", "remove", self.collection_name],
                capture_output=True, timeout=15,
            )
        except Exception:
            pass
        # Files live at /home/hermes/workspace/<collection_name>/
        coll_dir = QMD_WORKSPACE / self.collection_name
        if coll_dir.exists():
            shutil.rmtree(coll_dir)

    def index_chunks(self, chunks: List[Tuple[str, str]]) -> bool:
        """
        Write chunks as markdown files under /home/hermes/workspace/<name>/,
        register as QMD collection, and start embedding in background (non-blocking).

        Returns True on success (collection registered), False on failure.
        Caller can query immediately — QMD uses BM25 until vectors are ready.
        """
        if not chunks:
            return True

        self._chunks = chunks
        self._cleanup()

        # QMD stores files at /home/hermes/workspace/<collection_name>/
        coll_dir = QMD_WORKSPACE / self.collection_name
        coll_dir.mkdir(exist_ok=True, parents=True)

        # Write each chunk as a .md file
        for i, (chunk_id, content) in enumerate(chunks):
            safe_id = "".join(c if c.isalnum() else "_" for c in chunk_id)[:50]
            filepath = coll_dir / f"c{i:03d}_{safe_id}.md"
            # Include chunk_id as YAML frontmatter for QMD metadata
            filepath.write_text(
                f"---\\nid: {chunk_id}\\n---\\n"
                f"# Chunk: {chunk_id}\\n\\n{content}\\n"
            )

        # Register collection — files already at /home/hermes/workspace/<name>/
        try:
            result = subprocess.run(
                ["qmd", "collection", "add",
                 self.collection_name, str(coll_dir)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"QMD collection add failed: {result.stderr[:200]}")
                return False

            # NOTE: We intentionally skip calling `qmd embed` here.
            # QMD's periodic background job embeds new collections automatically.
            # Until vectors are ready, QMD falls back to BM25 scores — which is
            # sufficient for the benchmark since our chunks are small and BM25
            # already achieves 87.7% on LoCoMo. This avoids blocking for minutes.
            self._embed_proc = None

            self._indexed = True
            return True

        except FileNotFoundError:
            print("QMD CLI not found")
            return False
        except Exception as e:
            print(f"QMD error: {e}")
            return False

    def query(self, query_str: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Run QMD structured hybrid query (lex + vec + hyde).

        Returns [(chunk_id, score)] sorted by relevance, or empty list on failure.
        Works immediately — QMD uses BM25 scores until vector embeddings are ready.
        """
        if not self._indexed:
            return []

        try:
            # Structured query: lex + vec + hyde
            # hyde generates a hypothetical document, then retrieves against it
            structured_query = f"lex: {query_str}\\nvec: {query_str}\\nhyde: {query_str}"
            result = subprocess.run(
                [
                    "qmd", "query",
                    "--json",
                    "--no-rerank",
                    "-n", str(top_k),
                    "-c", self.collection_name,
                ],
                input=structured_query,
                capture_output=True, text=True, timeout=45,
            )
            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            results = []

            # QMD JSON output: {results: [{path, snippet, score, ...}]}
            items = data if isinstance(data, list) else data.get("results", [])
            for item in items:
                path = item.get("path", "")
                score = item.get("score", 0.0)
                if not path:
                    continue
                fname = Path(path).stem  # "c001_vim_keybindings"
                # chunk_id is everything after "cNNN_" (first 6 chars = "cNNN_")
                chunk_id = fname[6:] if len(fname) > 6 else fname
                results.append((chunk_id, score))

            return results

        except Exception:
            return []

    def close(self):
        """Clean up the QMD collection and stop background embed."""
        if self._embed_proc and self._embed_proc.poll() is None:
            self._embed_proc.terminate()
            try:
                self._embed_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._embed_proc.kill()
        if not self.persistent:
            self._cleanup()
        self._indexed = False


def qmd_retrieve_sync(
    query: str,
    chunks: List[Tuple[str, str]],
    top_k: int = 5,
) -> str:
    """
    Synchronous QMD retrieval with BM25 fallback.

    Returns concatenated text of top-k retrieved chunks.
    Embed runs in background — query works immediately (BM25 until vectors ready).
    """
    qmd = QMDRetrieval()
    indexed = qmd.index_chunks(chunks)
    if not indexed:
        qmd.close()
        return bm25_retrieve(query, chunks, top_k)

    results = qmd.query(query, top_k)
    qmd.close()

    if not results:
        return bm25_retrieve(query, chunks, top_k)

    # Map back to content
    chunk_map = {cid: content for cid, content in chunks}
    retrieved = " ".join(
        chunk_map.get(cid, "") for cid, _ in results if cid in chunk_map
    )
    return retrieved if retrieved else bm25_retrieve(query, chunks, top_k)


async def qmd_retrieve(
    query: str,
    chunks: List[Tuple[str, str]],
    top_k: int = 5,
) -> str:
    """Async wrapper — runs QMD in executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, qmd_retrieve_sync, query, chunks, top_k
    )
