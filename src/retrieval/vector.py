from typing import List

from langchain_core.documents import Document

from src.ingestion.indexer import load_vectorstore


class VectorRetriever:
    """
    Vector-based retriever using similarity search over embeddings.
    """

    def __init__(self, k: int = 5):
        """
        :param k: number of top documents to retrieve
        """
        self.k = k
        self.vectorstore = load_vectorstore()

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k relevant document chunks for a query.
        """
        results = self.vectorstore.similarity_search(
            query=query,
            k=self.k,
        )

        return results
