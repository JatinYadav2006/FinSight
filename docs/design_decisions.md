## ADR-001: Use Official SEC EDGAR Endpoints

### Decision
Use the official SEC APIs (`company_tickers.json` and `submissions/CIKxxxx.json`) instead of third-party libraries.

### Why
- Transparent implementation
- No external dependency
- Easier to explain in interviews
- Direct control over requests
- Easier to debug

### Alternatives Considered
- sec-api
- sec-edgar-downloader

### Reason for Rejection
They abstract away implementation details that are valuable for understanding and demonstrating the ingestion pipeline.