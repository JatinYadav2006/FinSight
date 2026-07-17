from src.ingestion.edgar_client import EDGARClient
from src.ingestion.document_cleaner import DocumentCleaner

client = EDGARClient()
cleaner = DocumentCleaner()

doc = cleaner.clean(client.get_latest_10k("AAPL"))

print(doc.clean_text[:2000])