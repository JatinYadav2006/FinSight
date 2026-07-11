"""
Client for interacting with the SEC EDGAR API.
"""

from requests import Session

from src.models import RawDocument
from src.ingestion.sec_constants import (
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    USER_AGENT,
    REQUEST_TIMEOUT_SECONDS,
)


class EDGARClient:
    """
    Client responsible for interacting with the SEC EDGAR API.

    This class downloads SEC filings and converts them into
    FinSight domain models.
    """

    def __init__(self) -> None:
        self.session: Session = Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_latest_10k(self, ticker: str) -> RawDocument:
        cik = self._get_cik(ticker)
        accession_number, filing_year = self._find_latest_10k(cik)
        raw_text = self._download_filing(cik, accession_number)
        return self._build_raw_document(ticker, filing_year, accession_number, raw_text)

    def _get_cik(self, ticker: str) -> str:
        raise NotImplementedError

    def _find_latest_10k(self, cik: str) -> tuple[str, int]:
        raise NotImplementedError

    def _download_filing(self, cik: str, accession_number: str) -> str:
        raise NotImplementedError

    def _build_raw_document(
        self,
        ticker: str,
        filing_year: int,
        accession_number: str,
        raw_text: str,
    ) -> RawDocument:
        raise NotImplementedError