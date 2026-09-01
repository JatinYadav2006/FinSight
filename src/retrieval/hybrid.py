"""
Hybrid retrieval using BM25 + dense retrieval + Reciprocal Rank Fusion.
"""

from __future__ import annotations

from collections import defaultdict

from src.models import Chunk, RetrievedChunk
from src.retrieval.bm25 import BM25Retriever, BM25Result
from src.retrieval.dense import DenseRetriever, DenseResult


RRF_K = 60
DEFAULT_DENSE_TOP_K = 15
DEFAULT_BM25_TOP_K = 15
DEFAULT_FINAL_TOP_K = 8


class HybridRetriever:
    """
    Combines BM25 and dense retrieval using Reciprocal Rank Fusion.

    Each retrieval path returns its own ranked results. RRF then combines
    those rankings without manually tuning score weights.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        company_ticker: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve and fuse BM25 + dense results.

        BM25 and dense retrieval each produce up to 15 candidates by
        default. RRF combines their rankings and returns the final
        top-k results.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        bm25_results = self._bm25.search(
            query,
            top_k=DEFAULT_BM25_TOP_K,
            company_ticker=company_ticker,
        )

        dense_results = self._dense.search(
            query,
            top_k=DEFAULT_DENSE_TOP_K,
            company_ticker=company_ticker,
        )

        return self._fuse(
            bm25_results=bm25_results,
            dense_results=dense_results,
            top_k=top_k,
        )

    @staticmethod
    def _fuse(
        bm25_results: list[BM25Result],
        dense_results: list[DenseResult],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Fuse ranked BM25 and dense results using Reciprocal Rank Fusion.
        """
        chunks_by_id: dict[str, Chunk] = {}

        bm25_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}

        dense_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}

        rrf_scores: defaultdict[str, float] = defaultdict(float)

        # ---------------------------------------------------------------
        # BM25 contribution
        # ---------------------------------------------------------------
        for result in bm25_results:
            chunk_id = result.chunk.id

            chunks_by_id[chunk_id] = result.chunk
            bm25_scores[chunk_id] = result.score
            bm25_ranks[chunk_id] = result.rank

            rrf_scores[chunk_id] += (
                1.0 / (RRF_K + result.rank)
            )

        # ---------------------------------------------------------------
        # Dense contribution
        # ---------------------------------------------------------------
        for result in dense_results:
            chunk_id = result.chunk.id

            chunks_by_id[chunk_id] = result.chunk
            dense_scores[chunk_id] = result.similarity_score
            dense_ranks[chunk_id] = result.rank

            rrf_scores[chunk_id] += (
                1.0 / (RRF_K + result.rank)
            )

        # ---------------------------------------------------------------
        # Sort by fused score.
        #
        # ID is used as a deterministic tie-breaker so identical RRF
        # scores always produce the same ordering.
        # ---------------------------------------------------------------
        ranked_ids = sorted(
            rrf_scores,
            key=lambda chunk_id: (
                -rrf_scores[chunk_id],
                chunk_id,
            ),
        )[:top_k]

        results: list[RetrievedChunk] = []

        for rrf_rank, chunk_id in enumerate(
            ranked_ids,
            start=1,
        ):
            results.append(
                RetrievedChunk(
                    chunk=chunks_by_id[chunk_id],
                    dense_similarity_score=dense_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    dense_rank=dense_ranks.get(
                        chunk_id,
                        0,
                    ),
                    bm25_score=bm25_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    bm25_rank=bm25_ranks.get(
                        chunk_id,
                        0,
                    ),
                    rrf_score=rrf_scores[chunk_id],
                    rrf_rank=rrf_rank,
                )
            )

        return results