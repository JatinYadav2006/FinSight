"""
BM25 lexical retrieval over a multi-company chunk corpus.
"""

from rank_bm25 import BM25Okapi

from src.models import Chunk
from src.retrieval.lexical import tokenize

from dataclasses import dataclass

from src.models import Chunk


@dataclass(frozen=True)
class BM25Result:
    chunk: Chunk
    score: float
    rank: int

class BM25Retriever:
    """
    Indexes the full chunk corpus across all target companies in one
    unified index (not per-company), so a single query can serve both
    single-company questions (narrowed via company_ticker) and
    cross-company comparison questions (left unfiltered) without
    maintaining separate index objects.

    Public API:
        retriever = BM25Retriever(chunks)
        results = retriever.search("Apple net sales 2025", top_k=10)
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")
        corpus = [tokenize(chunk.text) for chunk in chunks]

        empty = [c.id for c, tokens in zip(chunks, corpus) if not tokens]
        if empty:
            raise ValueError(
                f"{len(empty)} chunk(s) tokenized to zero terms, e.g. {empty[0]}. "
                "BM25 cannot meaningfully score an empty document."
            )

        self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        company_ticker: str | None = None,
    ) -> list[BM25Result]:
        """
        Return the highest-scoring chunks for a lexical query.

        When company_ticker is provided, only chunks belonging to that
        company are eligible for the ranking.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[BM25Result] = []

        for index in ranked_indices:
            chunk = self._chunks[index]

            if company_ticker is not None:
                if chunk.company_ticker.upper() != company_ticker.upper():
                    continue

            results.append(
                BM25Result(
                    chunk=chunk,
                    score=float(scores[index]),
                    rank=len(results) + 1,
                )
            )

            if len(results) == top_k:
                break

        return results