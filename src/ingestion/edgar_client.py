"""
Client for interacting with the SEC EDGAR API.
"""

from requests import Session
from datetime import date
from src.models import RawDocument
from src.ingestion.sec_constants import (
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    SEC_ARCHIVES_BASE_URL,
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
        self._ticker_mapping: dict[str, tuple[str, str]] | None = None

    def get_latest_10k(self, ticker: str) -> RawDocument:
        cik = self._get_cik(ticker)
        accession_number, primary_document, filing_year = self._find_latest_10k(cik)
        raw_text = self._download_filing(cik, accession_number, primary_document)
        return self._build_raw_document(
            ticker,
            filing_year,
            accession_number,
            primary_document,
            raw_text,
        )

    def _get_cik(self, ticker: str) -> str:
        ticker = ticker.upper()

        if self._ticker_mapping is None:
            self._load_ticker_mapping()

        if ticker not in self._ticker_mapping:
            raise ValueError(f"Unknown ticker: {ticker}")

        cik, _ = self._ticker_mapping[ticker]
        return cik

    def _get_company_name(self, ticker: str) -> str:
        """
        Retrieve the company name for a given ticker from the cached SEC mapping.
        """
        ticker = ticker.upper()

        if self._ticker_mapping is None:
            self._load_ticker_mapping()

        if ticker not in self._ticker_mapping:
            raise ValueError(f"Unknown ticker: {ticker}")

        _, company_name = self._ticker_mapping[ticker]
        return company_name

    # Download the official SEC ticker-to-CIK mapping.
    def _load_ticker_mapping(self) -> None:
        response = self.session.get(
            SEC_TICKERS_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        data = response.json()

        self._ticker_mapping = {
            entry["ticker"]: (
                str(entry["cik_str"]).zfill(CIK_PADDING),
                entry["title"],
            )
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

    def _download_filing(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        """
        Download the primary SEC filing document.
        Returns raw HTML exactly as received from EDGAR.
        """
        cik_directory = str(int(cik))
        accession_path = accession_number.replace("-", "")

        url = (
            f"{SEC_ARCHIVES_BASE_URL}/"
            f"{cik_directory}/"
            f"{accession_path}/"
            f"{primary_document}"
        )

        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        return response.text

    def _build_raw_document(
        self,
        ticker: str,
        filing_year: int,
        accession_number: str,
        primary_document: str,
        raw_text: str,
    ) -> RawDocument:
        """
        Construct a RawDocument domain object from downloaded filing content.
        """
        company_name = self._get_company_name(ticker)

        cik = self._get_cik(ticker)
        cik_directory = str(int(cik))
        accession_path = accession_number.replace("-", "")

        source_url = (
            f"{SEC_ARCHIVES_BASE_URL}/"
            f"{cik_directory}/"
            f"{accession_path}/"
            f"{primary_document}"
        )

        return RawDocument(
            company_ticker=ticker.upper(),
            company_name=company_name,
            filing_year=filing_year,
            filing_type="10-K",
            source_url=source_url,
            raw_text=raw_text,
        )