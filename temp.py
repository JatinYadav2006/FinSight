"""
Verifies that every extracted Section satisfies:

document.clean_text[section.char_start:section.char_end] == section.text

Run after fixing _build_section().
Delete after verification.
"""

from src.ingestion.edgar_client import EDGARClient
from src.ingestion.document_cleaner import DocumentCleaner
from src.ingestion.section_extractor import SectionExtractor

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]

client = EDGARClient()
cleaner = DocumentCleaner()
extractor = SectionExtractor()

for ticker in TICKERS:
    print("=" * 80)
    print(ticker)
    print("=" * 80)

    raw_document = client.get_latest_10k(ticker)
    clean_document = cleaner.clean(raw_document)
    sections = extractor.extract(clean_document)

    for section in sections:
        expected = clean_document.clean_text[
            section.char_start:section.char_end
        ]

        assert expected == section.text, (
            f"{ticker} | {section.section_name}\n"
            f"Section offsets do not match text.\n"
            f"Expected: {expected[:100]!r}\n"
            f"Actual:   {section.text[:100]!r}"
        )

        print(
            f"✓ {section.section_name:<35}"
            f" ({len(section.text):>7} chars)"
        )

    print()

print("=" * 80)
print("SUCCESS: All section offsets verified across all companies.")
print("=" * 80)