from pathlib import Path

from src.ingestion.load_directory import load_documents_from_dir
from src.ingestion.splitter import split_documents
from src.ingestion.indexer import index_documents, load_vectorstore
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from src.retrieval.hybrid import HybridRetriever

VECTOR_DIR = "data/vectorstore"
BM25_PATH = "data/bm25.pkl"
DOCS_DIR = "docs"


def build_or_load_retriever() -> HybridRetriever:
    vector_exists = Path(VECTOR_DIR).exists()
    bm25_exists = Path(BM25_PATH).exists()

    if vector_exists and bm25_exists:
        print("✅ Loading existing indices")
        vector = VectorRetriever(k=5)
        bm25 = BM25DocRetriever.load(BM25_PATH)
        return HybridRetriever(vector, bm25)

    print("⚙️ Building indices (first run or docs changed)")

    docs = load_documents_from_dir(DOCS_DIR)
    chunks = split_documents(docs)

    index_documents(chunks)

    bm25 = BM25DocRetriever(chunks)
    bm25.save(BM25_PATH)

    vector = VectorRetriever(k=5)
    return HybridRetriever(vector, bm25)