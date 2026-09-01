"""
Splits SEC filing sections into token-budgeted, sentence-aware chunks.
"""

import pysbd
import tiktoken

from dataclasses import dataclass
from src.models import Section, Chunk


TARGET_TOKENS = 448
OVERLAP_TOKENS = 64

# gpt-4o-mini uses o200k_base -- verified against tiktoken's own model
# mapping. Chunk sizes should reflect what the actual downstream model
# sees, not a generic or older encoding.
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

@dataclass(frozen=True)
class _PositionedSentence:
    """
    Internal representation of a sentence together with its exact
    character offsets within section.text.
    """

    text: str
    start: int
    end: int

class Chunker:
    """
    Splits a Section into sentence-aware, token-budgeted chunks with
    sentence-level overlap between consecutive chunks.

    Public API:
        chunker = Chunker()
        chunks = chunker.chunk(section)
    """

    def chunk(self, section: Section) -> list[Chunk]:
        """Public API: split a Section into token-budgeted, overlapping Chunks."""
        sentences = self._split_into_sentences(section.text)
        self._verify_reconstruction(section.text, sentences)
        positioned = self._position_sentences(sentences)
        groups = self._group_sentences(positioned)
        return self._build_chunks(section, groups)

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

    def _position_sentences(
        self,
        sentences: list[str],
    ) -> list[_PositionedSentence]:
        """
        Attach exact character offsets to each sentence using a running cursor.

        Because sentence reconstruction has already been verified,
        sentences form a contiguous partition of the original text.
        """
        positioned: list[_PositionedSentence] = []
        cursor = 0

        for sentence in sentences:
            end = cursor + len(sentence)

            positioned.append(
                _PositionedSentence(
                    text=sentence,
                    start=cursor,
                    end=end,
                )
            )

            cursor = end

        return positioned

    def _count_tokens(self, text: str) -> int:
        return len(TOKENIZER.encode(text))

    def _split_oversized_sentence(
        self, sentence: _PositionedSentence
    ) -> list[_PositionedSentence]:
        """
        Split an oversized "sentence" (flattened table/exhibit list) into
        token windows, with the same overlap step as normal chunks.

        Offsets must be derived by decoding the cumulative token prefix at
        each window boundary -- NOT by assuming a fixed characters-per-token
        ratio. Tokens don't map 1:1 to characters, so window i's character
        start is len(decode(token_ids[:i])), not i * (avg chars per token).
        Getting this wrong would silently misplace every chunk built from an
        oversized sentence.
        """
        token_ids = TOKENIZER.encode(sentence.text)
        step = TARGET_TOKENS - OVERLAP_TOKENS

        windows = []
        for i in range(0, len(token_ids), step):
            window_ids = token_ids[i : i + TARGET_TOKENS]
            window_text = TOKENIZER.decode(window_ids)
            prefix_char_length = len(TOKENIZER.decode(token_ids[:i]))

            window_start = sentence.start + prefix_char_length
            windows.append(
                _PositionedSentence(
                    text=window_text,
                    start=window_start,
                    end=window_start + len(window_text),
                )
            )
        return windows

    def _carry_overlap(
        self, previous_group: list[_PositionedSentence]
    ) -> tuple[list[_PositionedSentence], int]:
        """
        Walk backward from the previous group's tail until ~OVERLAP_TOKENS
        is covered. Capped: once overlap is non-empty, a candidate sentence
        larger than OVERLAP_TOKENS on its own is not added -- prevents an
        oversized sentence sitting earlier in the group from being swept in
        wholesale just because trailing sentences were too short to reach
        the target alone.
        """
        overlap: list[_PositionedSentence] = []
        overlap_tokens = 0
        for sentence in reversed(previous_group):
            sentence_tokens = self._count_tokens(sentence.text)
            if overlap and sentence_tokens > OVERLAP_TOKENS:
                break
            overlap.insert(0, sentence)
            overlap_tokens += sentence_tokens
            if overlap_tokens >= OVERLAP_TOKENS:
                break
        return overlap, overlap_tokens

    def _group_sentences(
        self, sentences: list[_PositionedSentence]
    ) -> list[list[_PositionedSentence]]:
        groups: list[list[_PositionedSentence]] = []
        current: list[_PositionedSentence] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence.text)

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

                # Keep this! Prevents reintroducing the MSFT overflow bug.
                if current_tokens + sentence_tokens > TARGET_TOKENS:
                    current = []
                    current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            groups.append(current)

        return groups

    def _build_chunks(
        self,
        section: Section,
        groups: list[list[_PositionedSentence]],
    ) -> list[Chunk]:
        """
        Convert grouped, positioned sentences into Chunk objects.

        Sentence positions are relative to section.text, so absolute
        document offsets are obtained by adding section.char_start.
        """
        total_chunks = len(groups)
        chunks: list[Chunk] = []

        for index, group in enumerate(groups):
            text = "".join(sentence.text for sentence in group)

            chunk_id = (
                f"{section.company_ticker}_"
                f"{section.filing_year}_"
                f"{section.section_id.value}_"
                f"{index:03d}"
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    company_ticker=section.company_ticker,
                    company_name=section.company_name,
                    filing_year=section.filing_year,
                    filing_type=section.filing_type,
                    section_id=section.section_id,
                    section_name=section.section_name,
                    text=text,
                    char_start=section.char_start + group[0].start,
                    char_end=section.char_start + group[-1].end,
                    token_count=self._count_tokens(text),
                    chunk_index=index,
                    total_chunks=total_chunks,
                )
            )

        return chunks