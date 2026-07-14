"""
Client for interacting with the SEC EDGAR API.
"""

from requests import Session
from datetime import date
from src.models import RawDocument
from src.ingestion.sec_constants import (
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    USER_AGENT,
    REQUEST_TIMEOUT_SECONDS,
    CIK_PADDING,
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
        self._ticker_mapping: dict[str, str] | None = None

    def get_latest_10k(self, ticker: str) -> RawDocument:
        cik = self._get_cik(ticker)
        accession_number, primary_document, filing_year = self._find_latest_10k(cik)
        raw_text = self._download_filing(cik, accession_number, primary_document)
        return self._build_raw_document(ticker, filing_year, accession_number, raw_text)

    def _get_cik(self, ticker: str) -> str:
        ticker = ticker.upper()

        if self._ticker_mapping is None:
            self._load_ticker_mapping()

        cik = self._ticker_mapping.get(ticker)
        if cik is None:
            raise ValueError(f"Unknown ticker: {ticker}")

        return cik
    # Download the official SEC ticker-to-CIK mapping.
    def _load_ticker_mapping(self) -> None:
        response = self.session.get(    
            SEC_TICKERS_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        data = response.json()

        self._ticker_mapping = {
            entry["ticker"]: str(entry["cik_str"]).zfill(CIK_PADDING)
            for entry in data.values()
        }

    def _find_latest_10k(self, cik: str) -> tuple[str, str, int]:
        url = SEC_SUBMISSIONS_URL.format(cik=cik)

        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        data = response.json()
        recent_filings = data["filings"]["recent"]

        forms = recent_filings["form"]
        accession_numbers = recent_filings["accessionNumber"]
        primary_documents = recent_filings["primaryDocument"]
        filing_dates = recent_filings["filingDate"]

        for index, form in enumerate(forms):
            if form == "10-K":
                accession_number = accession_numbers[index]
                primary_document = primary_documents[index]
                filing_date = filing_dates[index]
                filing_year = date.fromisoformat(filing_date).year
                return accession_number, primary_document, filing_year

        raise ValueError(f"No 10-K filing found for CIK {cik}")

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