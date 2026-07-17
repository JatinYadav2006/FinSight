"""
Constants for interacting with the SEC EDGAR API.
"""

SEC_BASE_URL = "https://www.sec.gov"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

USER_AGENT = "FinSight/0.1 yadavjatin2006@gmail.com"

REQUEST_TIMEOUT_SECONDS = 30

CIK_PADDING = 10