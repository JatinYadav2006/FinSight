from dataclasses import dataclass
from enum import StrEnum


class SectionID(StrEnum):
    """
    Controlled vocabulary for supported SEC filing sections.
    Ensures type safety and prevents typos across the pipeline.
    """
    ITEM_1 = "Item_1"
    ITEM_1A = "Item_1A"
    ITEM_7 = "Item_7"
    ITEM_8 = "Item_8"


@dataclass(frozen=True)
class RawDocument:
    """
    Represents an SEC filing exactly as downloaded from EDGAR.
    No cleaning or preprocessing has been applied.
    """

    company_ticker: str
    company_name: str
    filing_year: int
    filing_type: str
    source_url: str
    raw_text: str


@dataclass(frozen=True)
class CleanDocument:
    """
    Represents a cleaned SEC filing ready for section extraction.
    """

    company_ticker: str
    company_name: str
    filing_year: int
    filing_type: str
    clean_text: str


@dataclass(frozen=True)
class Section:
    """
    Represents a single logical SEC filing section prior to chunking.
    """

    company_ticker: str
    company_name: str
    filing_year: int
    filing_type: str

    section_id: SectionID
    section_name: str

    text: str

    char_start: int
    char_end: int


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Describes the origin and location of a chunk within an SEC filing.
    """

    company_ticker: str
    company_name: str
    filing_year: int
    filing_type: str

    section_id: SectionID
    section_name: str

    chunk_index_in_section: int
    total_chunks_in_section: int

    char_start: int
    char_end: int


@dataclass(frozen=True)
class Chunk:
    """
    Represents a retrievable unit of knowledge from an SEC filing.
    """

    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Represents the retrieval metadata associated with a chunk
    returned by the hybrid retrieval pipeline.
    """

    chunk: Chunk

    dense_similarity_score: float
    dense_rank: int

    bm25_score: float
    bm25_rank: int

    rrf_score: float
    rrf_rank: int