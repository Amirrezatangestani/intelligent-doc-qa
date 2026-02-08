from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from PIL import Image
import pytesseract


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}


def load_document(file_path: str) -> List[Document]:
    """
    Load a document and return a list of LangChain Document objects.
    Supports PDF, TXT, DOCX, and Images (OCR).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    if path.suffix.lower() == ".txt":
        return _load_txt(path)

    if path.suffix.lower() == ".docx":
        return _load_docx(path)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return _load_image(path)

    raise RuntimeError("Unhandled file type")


# -------------------- LOADERS -------------------- #

def _load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    # ensure metadata consistency
    for d in docs:
        d.metadata["source"] = path.name

    return docs


def _load_txt(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = path.name

    return docs


def _load_docx(path: Path) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(str(path))
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = path.name

    return docs


def _load_image(path: Path) -> List[Document]:
    image = Image.open(path)
    text = pytesseract.image_to_string(image)

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "type": "image_ocr"
            }
        )
    ]