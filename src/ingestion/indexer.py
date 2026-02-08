from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


PERSIST_DIR = "data/vectorstore"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def index_documents(
    chunks: List[Document],
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Create or update a Chroma vector store from document chunks.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    vectorstore.persist()
    return vectorstore


def load_vectorstore(
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    """
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    
print("✅ Indexing finished")
