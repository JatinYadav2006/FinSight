"""
Extracts target SEC sections from a cleaned filing.
"""

import re
from dataclasses import dataclass
from collections import defaultdict

from src.models import CleanDocument, Section

CONTEXT_WINDOW = 60
PRECEDING_TEXT_WINDOW = 5
HEADING_PUNCTUATION = (".", ":")

NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]")
QUOTE_CHARS = ('"', "\u201c", "\u201d")

TITLE_KEYWORDS = {
    "Item 1": "business",
    "Item 1A": "riskfactors",
    "Item 7": "managementsdiscussion",
    "Item 7A": "quantitativeandqualitative",
    "Item 8": "financialstatements",
}

ITEM_PATTERNS = {
    "Item 1": re.compile(r"Item\s+1(?![0-9A-Za-z])", re.IGNORECASE),
    "Item 1A": re.compile(r"Item\s+1A(?![0-9A-Za-z])", re.IGNORECASE),
    "Item 7": re.compile(r"Item\s+7(?![0-9A-Za-z])", re.IGNORECASE),
    "Item 7A": re.compile(r"Item\s+7A(?![0-9A-Za-z])", re.IGNORECASE),
    "Item 8": re.compile(r"Item\s+8(?![0-9A-Za-z])", re.IGNORECASE),
}


@dataclass(frozen=True)
class HeadingCandidate:
    """
    Represents a single candidate SEC section heading found within a
    cleaned filing. Internal parsing artifact -- not part of FinSight's
    public domain model.
    """

    label: str
    start: int
    end: int
    preceding_text: str
    following_text: str


class SectionExtractor:
    """
    Extracts target SEC sections from a cleaned filing.

    Public API:
        extractor = SectionExtractor()
        sections = extractor.extract(clean_document)
    """

    def extract(self, document: CleanDocument) -> list[Section]:
        """Extract all supported sections from a cleaned SEC filing."""
        candidates = self._find_heading_candidates(document.clean_text)
        heading_candidates = self._filter_heading_candidates(candidates)
        boundaries = self._select_section_boundaries(heading_candidates)
        return self._build_sections(document, boundaries)

    def _find_heading_candidates(self, text: str) -> list[HeadingCandidate]:
        """Scan the cleaned text for every regex match of every target label."""
        candidates: list[HeadingCandidate] = []
        for label, pattern in ITEM_PATTERNS.items():
            for m in pattern.finditer(text):
                candidates.append(
                    HeadingCandidate(
                        label=label,
                        start=m.start(),
                        end=m.end(),
                        preceding_text=text[max(0, m.start() - PRECEDING_TEXT_WINDOW):m.start()],
                        following_text=text[m.end():m.end() + CONTEXT_WINDOW],
                    )
                )
        return candidates

    def _normalize_heading_text(self, text: str) -> str:
        """Strip everything except letters/digits for keyword comparison."""
        return NON_ALPHANUMERIC_PATTERN.sub("", text.lower())

    def _filter_heading_candidates(
        self, candidates: list[HeadingCandidate]
    ) -> list[HeadingCandidate]:
        """
        Keep only candidates that are structurally real headings:

          1. A period or colon immediately follows the item label (before
             any other text). Verified against all 25 real headings across
             all 5 target companies -- every single one has this punctuation
             directly after the label. This is what separates a heading
             ("Item 1. Business") from a cross-reference embedded in prose
             ("Item 1 Business and Note 15 of the Notes...") -- both can
             contain the keyword "business" shortly after the label, but
             only the real heading has punctuation directly adjacent.
          2. The expected title keyword appears in the normalized text
             immediately after that punctuation.
          3. No opening quote mark sits directly before the label -- catches
             inline quoted cross-references like 'Refer to"Item 1A. Risk
             Factors"...' where the whole phrase is quoted even though it
             also has a period.

        Does not yet distinguish TOC entries from real headings -- both
        satisfy all three checks here. That's _select_section_boundaries()'s
        job.
        """
        heading_like = []
        for candidate in candidates:
            if any(q in candidate.preceding_text[-3:] for q in QUOTE_CHARS):
                continue

            stripped_start = candidate.following_text.lstrip()

            if stripped_start[:1] not in HEADING_PUNCTUATION:
                continue

            normalized = self._normalize_heading_text(stripped_start)
            keyword = TITLE_KEYWORDS[candidate.label]
            if keyword in normalized:
                heading_like.append(candidate)

        return heading_like

    def _select_section_boundaries(
        self, candidates: list[HeadingCandidate]
    ) -> dict[str, HeadingCandidate]:
        """
        For each label, select the candidate that sits farthest from its
        nearest neighbor (of any label) in the full sorted candidate list.
        TOC entries cluster tightly together with other labels' TOC
        entries; real headings sit far from any other surviving candidate.
        """
        if not candidates:
            raise ValueError("No heading candidates survived filtering.")

        sorted_candidates = sorted(candidates, key=lambda c: c.start)
        positions = [c.start for c in sorted_candidates]

        def isolation(index: int) -> float:
            distances = []
            if index > 0:
                distances.append(positions[index] - positions[index - 1])
            if index < len(positions) - 1:
                distances.append(positions[index + 1] - positions[index])
            return min(distances) if distances else float("inf")

        grouped: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(sorted_candidates):
            grouped[c.label].append(i)

        boundaries = {}
        for label in ITEM_PATTERNS:
            indices = grouped.get(label)
            if not indices:
                raise ValueError(f"No surviving heading candidate for {label}.")
            best_index = max(indices, key=isolation)
            boundaries[label] = sorted_candidates[best_index]

        return boundaries

    def _build_sections(self, document: CleanDocument, boundaries) -> list[Section]:
        raise NotImplementedError