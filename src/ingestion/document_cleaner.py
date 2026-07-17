"""
Converts RawDocument HTML/iXBRL content into clean plain text CleanDocument.
"""

import re

from bs4 import BeautifulSoup

from src.models import RawDocument, CleanDocument


NOISE_TAGS = ("script", "style","ix:header")
XML_DECLARATION_PATTERN = re.compile(r"^\s*<\?xml.*?\?>", re.IGNORECASE)
ALL_WHITESPACE_PATTERN = re.compile(r"\s+")
GET_TEXT_SEPARATOR = " "


class DocumentCleaner:
    """
    Converts a RawDocument into a CleanDocument by removing HTML/iXBRL
    markup while preserving all visible text content.

    Public API:
        cleaner = DocumentCleaner()
        clean_document = cleaner.clean(raw_document)
    """

    def clean(self, document: RawDocument) -> CleanDocument:
        """Transform a RawDocument's raw HTML into a readable CleanDocument."""
        stripped_text = self._strip_xml_declaration(document.raw_text)
        soup = self._parse_html(stripped_text)
        self._remove_noise_elements(soup)
        visible_text = self._extract_visible_text(soup)
        clean_text = self._normalize_whitespace(visible_text)

        if not clean_text.strip():
            raise ValueError(
                f"Cleaning produced empty text for {document.company_ticker} "
                f"{document.filing_year} 10-K. Raw document may be malformed."
            )

        return CleanDocument(
            company_ticker=document.company_ticker,
            company_name=document.company_name,
            filing_year=document.filing_year,
            filing_type=document.filing_type,
            clean_text=clean_text,
        )

    def _strip_xml_declaration(self, raw_text: str) -> str:
        """Remove the leading XML declaration line, if present."""
        return XML_DECLARATION_PATTERN.sub("", raw_text, count=1)

    def _parse_html(self, html_text: str) -> BeautifulSoup:
        """Parse HTML text into a navigable BeautifulSoup tree."""
        return BeautifulSoup(html_text, "html.parser")

    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Strip script and style elements, which never carry retrievable content."""
        for tag_name in NOISE_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

    def _extract_visible_text(self, soup: BeautifulSoup) -> str:
        """
        Extract visible text from the parsed tree using a space separator.

        A space (not newline) separator is deliberate: SEC iXBRL wraps
        individual numbers inline within running sentences via tags like
        <ix:nonFraction>. A newline separator would fragment those sentences
        one token per line. Paragraph structure is not preserved in V1 --
        SectionExtractor locates sections by substring search, not layout.
        """
        return soup.get_text(separator=GET_TEXT_SEPARATOR)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse all runs of whitespace (spaces, tabs, newlines) into a single space."""
        return ALL_WHITESPACE_PATTERN.sub(" ", text).strip()