"""
BM25-based retrieval for the CircAIdian benchmark harness.

BM25 (Okapi Best Matching 25) is a probabilistic relevance ranking algorithm
used in information retrieval. It handles:
- Term frequency saturation (diminishing returns for repeated terms)
- Document length normalization
- Rare term weighting (IDF)

This replaces simple keyword overlap for more robust retrieval.
"""
import math
from collections import Counter
from typing import List, Tuple


class BM25:
    """BM25 ranker for a corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.avgdl = 0.0
        self.doc_len: List[int] = []
        self.N = 0  # total docs
        self.doc_freqs: Counter = Counter()  # term -> num docs containing term

    def index(self, doc_ids: List[str], documents: List[str]):
        """Build the BM25 index from documents. Each document is a string."""
        self.doc_ids = doc_ids
        self.N = len(documents)
        self.documents = [self._tokenize(d) for d in documents]
        self.doc_len = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0

        # Compute document frequencies
        for doc in self.documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        return [
            w.lower().strip(".,!?:;\"'()[]{}")
            for w in text.split()
            if len(w) > 1
        ]

    def _idf(self, term: str) -> float:
        """Compute IDF for a term. N/df where df = document frequency."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard IDF formula with smoothing
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against one document."""
        query_terms = self._tokenize(query)
        doc = self.documents[doc_idx]
        doc_len = self.doc_len[doc_idx]
        score = 0.0

        tf = Counter(doc)
        for term in query_terms:
            if term not in tf:
                continue
            idf = self._idf(term)
            # BM25 term frequency component
            tf_component = (tf[term] * (self.k1 + 1)) / (
                tf[term] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )
            score += idf * tf_component

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a query. Returns [(doc_id, score)]."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[i], score) for i, score in scores[:top_k] if score > 0]


def bm25_retrieve(query: str, chunks: List[Tuple[str, str]], top_k: int = 5) -> str:
    """
    Retrieve the most relevant chunks for a query using BM25.

    Args:
        query: The question/query string
        chunks: List of (chunk_id, chunk_text) tuples
        top_k: Number of top chunks to return

    Returns:
        Concatenated text of top-k relevant chunks
    """
    if not chunks:
        return ""

    doc_ids = [cid for cid, _ in chunks]
    doc_texts = [text for _, text in chunks]

    ranker = BM25(k1=1.5, b=0.75)
    ranker.index(doc_ids, doc_texts)

    results = ranker.retrieve(query, top_k=top_k)
    retrieved_ids = [doc_id for doc_id, _ in results]

    # Return concatenated relevant chunks (in order of relevance)
    id_to_text = {cid: text for cid, text in chunks}
    return " ".join(id_to_text[cid] for cid in retrieved_ids if cid in id_to_text)
