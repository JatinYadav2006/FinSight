"""
Persistent ChromaDB index construction for FinSight.

The index stores:
- chunk IDs
- chunk text
- BGE-large embeddings
- retrieval/citation metadata

The canonical Chunk objects remain outside ChromaDB.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.models import Chunk
from src.retrieval.embedding import EmbeddingModel


CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "finsight_chunks"

EMBEDDING_BATCH_SIZE = 32


class ChromaIndexer:
    """
    Builds and persists the FinSight dense vector index.

    Public API:
        indexer = ChromaIndexer()
        indexer.build(chunks)
    """

    def __init__(
        self,
        persist_path: Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._embedding_model = embedding_model or EmbeddingModel()

        self._persist_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(
            path=str(self._persist_path),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build(
        self,
        chunks: list[Chunk],
    ) -> int:
        """
        Build the persistent ChromaDB collection from the supplied chunks.

        Existing entries with the same IDs are replaced, making repeated
        index builds deterministic and avoiding duplicate documents.
        """
        if not chunks:
            raise ValueError(
                "Cannot build ChromaDB index from an empty chunk list."
            )

        self._validate_chunks(chunks)

        total_added = 0

        for start in range(
            0,
            len(chunks),
            EMBEDDING_BATCH_SIZE,
        ):
            batch = chunks[
                start:start + EMBEDDING_BATCH_SIZE
            ]

            texts = [
                chunk.text
                for chunk in batch
            ]

            embeddings = self._embedding_model.embed_documents(
                texts
            )

            self._collection.upsert(
                ids=[
                    chunk.id
                    for chunk in batch
                ],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    self._metadata_for(chunk)
                    for chunk in batch
                ],
            )

            total_added += len(batch)

        return total_added

    def count(self) -> int:
        """Return the number of entries currently stored."""
        return self._collection.count()

    def _validate_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Validate chunk IDs and required text before indexing."""
        seen_ids: set[str] = set()

        for chunk in chunks:
            if not chunk.id:
                raise ValueError(
                    "Chunk ID must not be empty."
                )

            if chunk.id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID detected: {chunk.id}"
                )

            if not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.id} contains empty text."
                )

            seen_ids.add(chunk.id)

    @staticmethod
    def _metadata_for(
        chunk: Chunk,
    ) -> dict[str, str | int]:
        """Convert Chunk provenance into Chroma-compatible metadata."""
        return {
            "company_ticker": chunk.company_ticker,
            "company_name": chunk.company_name,
            "filing_year": chunk.filing_year,
            "filing_type": chunk.filing_type,
            "section_id": chunk.section_id.value,
            "section_name": chunk.section_name,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_count": chunk.token_count,
        }