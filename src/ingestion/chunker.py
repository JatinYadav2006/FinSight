"""
Splits SEC filing sections into token-budgeted, sentence-aware chunks.
"""

import pysbd
import tiktoken

from src.models import Section


TARGET_TOKENS = 512
OVERLAP_TOKENS = 64

# gpt-4o-mini uses o200k_base -- verified against tiktoken's own model
# mapping. Chunk sizes should reflect what the actual downstream model
# sees, not a generic or older encoding.
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


class Chunker:
    """
    Splits a Section into sentence-aware, token-budgeted chunks with
    sentence-level overlap between consecutive chunks.

    Public API:
        chunker = Chunker()
        chunks = chunker.chunk(section)
    """

    def chunk(self, section: Section) -> list[str]:
        raise NotImplementedError

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        clean=False preserves each sentence's exact original substring.
        Required for cursor-based offset tracking -- see
        _verify_reconstruction.
        """
        return SENTENCE_SEGMENTER.segment(text)

    def _verify_reconstruction(self, original: str, sentences: list[str]) -> None:
        """
        Precondition check: sentences must concatenate back into the
        original text exactly, since absolute character offsets rely on
        advancing a cursor through them in order rather than searching for
        each sentence's position (unreliable given how often financial
        tables repeat identical substrings).
        """
        reconstructed = "".join(sentences)
        if reconstructed != original:
            raise ValueError(
                "Sentence segmentation did not reconstruct the original "
                "section text exactly -- offset tracking would be unreliable. "
                f"Original length: {len(original)}, "
                f"reconstructed length: {len(reconstructed)}."
            )

    def _count_tokens(self, text: str) -> int:
        return len(TOKENIZER.encode(text))

    def _split_oversized_sentence(self, sentence: str) -> list[str]:
        """
        Some pysbd "sentences" aren't prose -- flattened financial tables
        or exhibit lists with no internal punctuation, sometimes 1000+
        tokens. Split by token windows rather than forcing an oversized
        chunk or truncating financial data.

        Uses the same step (TARGET_TOKENS - OVERLAP_TOKENS) as normal
        chunk boundaries, so windows within an oversized sentence overlap
        exactly as much as chunks elsewhere in the pipeline do -- no
        special case where overlap silently disappears just because the
        source happened to be one giant "sentence" instead of several
        normal ones.
        """
        token_ids = TOKENIZER.encode(sentence)
        step = TARGET_TOKENS - OVERLAP_TOKENS
        return [
            TOKENIZER.decode(token_ids[i:i + TARGET_TOKENS])
            for i in range(0, len(token_ids), step)
        ]

    def _carry_overlap(self, previous_group: list[str]) -> tuple[list[str], int]:
        """
        Build the start of the next group from the tail of the previous one,
        walking backward until ~OVERLAP_TOKENS worth of sentences are covered.

        Guards against unbounded duplication: once overlap already contains
        at least one sentence, a candidate sentence that alone exceeds
        OVERLAP_TOKENS is NOT added -- doing so would mean sweeping an
        oversized, unrelated sentence into "overlap" just because the trailing
        sentences of the previous group were too short to reach the target on
        their own (observed on AAPL Financial Statements: a 258-token group
        of three short sentences caused the entire group, including a
        241-token heading sentence, to be duplicated forward, overflowing the
        next chunk to 641 tokens). The first candidate is always included even
        if it alone exceeds OVERLAP_TOKENS -- overlap should never be empty,
        and a single oversized sentence at the tail is a legitimate, bounded
        case (unlike sweeping in an unrelated large sentence further back).
        """
        overlap: list[str] = []
        overlap_tokens = 0
        for sentence in reversed(previous_group):
            sentence_tokens = self._count_tokens(sentence)

            if overlap and overlap_tokens + sentence_tokens > OVERLAP_TOKENS:
                break

            overlap.insert(0, sentence)
            overlap_tokens += sentence_tokens

            if overlap_tokens >= OVERLAP_TOKENS:
                break
        return overlap, overlap_tokens

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        """
        Accumulate sentences into token-budgeted groups with sentence-level
        overlap carried into the next group. An oversized single sentence
        is emitted as its own group(s) via _split_oversized_sentence.
        """
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if sentence_tokens > TARGET_TOKENS:
                if current:
                    groups.append(current)
                for window in self._split_oversized_sentence(sentence):
                    groups.append([window])
                current, current_tokens = [], 0
                continue

            if current and current_tokens + sentence_tokens > TARGET_TOKENS:
                groups.append(current)
                current, current_tokens = self._carry_overlap(current)

                # If the overlap itself leaves no room for the incoming sentence,
                # drop the overlap for this boundary rather than exceeding the
                # maximum chunk size.
                if current_tokens + sentence_tokens > TARGET_TOKENS:
                    current = []
                    current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            groups.append(current)

        return groups

    def _build_chunks(self, section: Section, sentence_groups: list[list[str]]):
        raise NotImplementedError  # pending Chunk/ChunkMetadata field confirmation