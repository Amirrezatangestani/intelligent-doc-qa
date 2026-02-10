from src.ingestion.loaders import load_document
from src.ingestion.splitter import split_documents
from src.ingestion.indexer import index_documents

files = [
    "docs/sample.pdf",
    "docs/sample.txt",
    "docs/sample.docx",
]

all_docs = []
for f in files:
    docs = load_document(f)
    all_docs.extend(docs)

chunks = split_documents(all_docs)

index_documents(chunks)

print("✅ Documents indexed")
