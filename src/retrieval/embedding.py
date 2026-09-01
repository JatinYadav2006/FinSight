"""
BGE-large embedding model for FinSight dense retrieval.

Queries and documents intentionally use separate methods because
BGE uses asymmetric retrieval encoding:
- documents/passages are embedded as-is
- queries receive the documented retrieval instruction
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingModel:
    """
    Local BGE-large embedding model.

    Public API:
        embed_documents(texts)
        embed_query(text)
    """

    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME)

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Embed document/passages without a query instruction.
        """
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not contain empty strings.")

        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Embed a search query using BGE's retrieval instruction.
        """
        if not text.strip():
            raise ValueError("Query text must not be empty.")

        return self._model.encode(
            QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )