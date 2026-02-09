from src.ingestion.loaders import load_document
from src.ingestion.splitter import split_documents
from src.ingestion.indexer import index_documents

files = [
    "test_files/sample.pdf",
    "test_files/sample.txt",
    "test_files/sample.docx",
]

all_docs = []
for f in files:
    docs = load_document(f)
    all_docs.extend(docs)

chunks = split_documents(all_docs)

index_documents(chunks)

print("✅ Documents indexed")
