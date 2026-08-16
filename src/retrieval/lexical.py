"""
Lexical tokenization for BM25 indexing.

Isolated from bm25.py so the tokenizer can be swapped or refined later
(e.g. to preserve "10-K", percentages, dollar figures as coherent units)
without touching retrieval architecture -- verified against real AAPL
chunks in Gate 5.1's library check before this was written.
"""

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())