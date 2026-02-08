from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into overlapping chunks while preserving metadata.
    Adds a unique chunk_id to each chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_docs: List[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)

        for idx, chunk in enumerate(splits):
            metadata = dict(doc.metadata) if doc.metadata else {}

            # build stable chunk id
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "na")

            metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    return chunked_docs