"""
QMD-backed retrieval for CircAIdian benchmark harness.

Uses QMD's hybrid lex + vec + hyde retrieval via CLI subprocess.
Falls back to local BM25 if QMD is unavailable or fails.

Usage:
    python -c "from benchmark.qmd_retrieval import QMDRetrieval; ..."
"""
import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

from benchmark.bm25 import bm25_retrieve


# Workspace subdirectory for QMD benchmark collections
QMD_WORKSPACE = Path("/home/hermes/workspace/circadian-qmd")
QMD_WORKSPACE.mkdir(exist_ok=True)


class QMDRetrieval:
    """
    Manages a QMD collection for benchmark chunks with BM25 fallback.

    The workflow:
    1. index_chunks() — write chunks as .md files, add to QMD, embed
    2. query()         — run hybrid search, return (chunk_id, score) pairs
    3. close()         — clean up collection
    """

    def __init__(self, collection_name: str = "ca-bench"):
        self.collection_name = collection_name
        self._indexed = False
        self._chunks: List[Tuple[str, str]] = []

    def _cleanup(self):
        """Remove collection and local files."""
        try:
            subprocess.run(
                ["qmd", "collection", "remove", self.collection_name],
                capture_output=True, timeout=15,
            )
        except Exception:
            pass
        # Files are in the workspace dir; remove the whole collection subdir
        coll_dir = QMD_WORKSPACE / self.collection_name
        if coll_dir.exists():
            shutil.rmtree(coll_dir)

    def index_chunks(self, chunks: List[Tuple[str, str]]) -> bool:
        """
        Write chunks as markdown files, register as QMD collection, embed.

        Returns True on success, False on any failure (caller should use BM25).
        """
        if not chunks:
            return True

        self._chunks = chunks
        self._cleanup()

        # Create collection directory within workspace
        coll_dir = QMD_WORKSPACE / self.collection_name
        coll_dir.mkdir(exist_ok=True)

        # Write each chunk as a .md file
        for i, (chunk_id, content) in enumerate(chunks):
            safe_id = "".join(c if c.isalnum() else "_" for c in chunk_id)[:50]
            filepath = coll_dir / f"c{i:03d}_{safe_id}.md"
            # Include chunk_id as YAML frontmatter for QMD metadata
            filepath.write_text(
                f"---\nid: {chunk_id}\n---\n"
                f"# Chunk: {chunk_id}\n\n{content}\n"
            )

        # Register collection — QMD requires the workspace path
        try:
            result = subprocess.run(
                ["qmd", "collection", "add",
                 self.collection_name, str(coll_dir)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"QMD collection add failed: {result.stderr[:200]}")
                return False

            # Generate embeddings for vector search
            result = subprocess.run(
                ["qmd", "embed", "--max-docs-per-batch", "50"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"QMD embed failed: {result.stderr[:200]}")
                return False

            self._indexed = True
            return True

        except FileNotFoundError:
            print("QMD CLI not found")
            return False
        except subprocess.TimeoutExpired:
            print("QMD timed out")
            return False
        except Exception as e:
            print(f"QMD error: {e}")
            return False

    def query(self, query_str: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Run QMD structured hybrid query (lex + vec + hyde).

        Returns [(chunk_id, score)] sorted by relevance, or empty list on failure.
        """
        if not self._indexed:
            return []

        try:
            # Structured query: lex + vec + hyde
            # hyde generates a hypothetical document, then retrieves against it
            structured_query = f"lex: {query_str}\nvec: {query_str}\nhyde: {query_str}"
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
        """Clean up the QMD collection."""
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
