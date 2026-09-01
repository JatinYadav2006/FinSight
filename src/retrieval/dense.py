"""
Dense vector retrieval over the persistent FinSight ChromaDB index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.models import Chunk
from src.retrieval.embedding import EmbeddingModel


CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "finsight_chunks"

DEFAULT_TOP_K = 15


@dataclass(frozen=True)
class DenseResult:
    """
    Represents a chunk returned by dense retrieval.

    The original Chunk remains the canonical source of truth.
    """

    chunk: Chunk
    similarity_score: float
    rank: int


class DenseRetriever:
    """
    Retrieves semantically similar FinSight chunks from ChromaDB.

    Public API:
        retriever = DenseRetriever()
        results = retriever.search(
            "What was Apple's revenue in 2025?",
            top_k=15,
            company_ticker="AAPL",
        )
    """

    def __init__(
        self,
        persist_path: Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._embedding_model = embedding_model or EmbeddingModel()

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
        )

        try:
            self._collection = self._client.get_collection(
                name=collection_name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' "
                "could not be loaded. Build the index first."
            ) from exc

        if self._collection.count() == 0:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' is empty."
            )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company_ticker: str | None = None,
    ) -> list[DenseResult]:
        """
        Return the highest-ranked dense retrieval results.

        When company_ticker is provided, results are restricted to that
        company before ranking is returned.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Chroma needs enough candidates to apply the optional company
        # filter. For a filtered search, request the full collection and
        # filter locally. This is appropriate for FinSight's small corpus
        # and keeps the behavior deterministic.
        n_results = self._collection.count()

        query_embedding = self._embedding_model.embed_query(
            query
        )

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        ranked_results: list[DenseResult] = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            if company_ticker is not None:
                result_ticker = str(
                    metadata["company_ticker"]
                )

                if result_ticker.upper() != company_ticker.upper():
                    continue

            chunk = self._chunk_from_chroma(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
            )

            # Chroma returns cosine distance for our collection.
            # cosine similarity = 1 - cosine distance.
            similarity = 1.0 - float(distance)

            ranked_results.append(
                DenseResult(
                    chunk=chunk,
                    similarity_score=similarity,
                    rank=len(ranked_results) + 1,
                )
            )

            if len(ranked_results) == top_k:
                break

        return ranked_results

    @staticmethod
    def _chunk_from_chroma(
        chunk_id: str,
        document: str,
        metadata: dict,
    ) -> Chunk:
        """
        Reconstruct the canonical Chunk representation from Chroma data.
        """
        from src.models import SectionID

        return Chunk(
            id=chunk_id,
            company_ticker=str(metadata["company_ticker"]),
            company_name=str(metadata["company_name"]),
            filing_year=int(metadata["filing_year"]),
            filing_type=str(metadata["filing_type"]),
            section_id=SectionID(str(metadata["section_id"])),
            section_name=str(metadata["section_name"]),
            text=document,
            char_start=int(metadata["char_start"]),
            char_end=int(metadata["char_end"]),
            token_count=0,
            chunk_index=int(metadata["chunk_index"]),
            total_chunks=int(metadata["total_chunks"]),
        )