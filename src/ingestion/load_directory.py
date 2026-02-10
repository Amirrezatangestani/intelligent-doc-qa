from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.ingestion.loaders import load_document, SUPPORTED_EXTENSIONS


def load_documents_from_dir(dir_path: str) -> List[Document]:
    docs: List[Document] = []
    path = Path(dir_path)

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid documents directory: {dir_path}")

    for file in path.iterdir():
        if file.name.startswith("~$"):
            continue
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_document(str(file)))

    return docs