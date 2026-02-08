from typing import List
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from langchain_core.documents import Document


class HybridRetriever:
    """
    Combine Vector Retriever and BM25 Retriever results.
    """
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25DocRetriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1️⃣ Vector results
        vector_docs = self.vector_retriever.retrieve(query)

        # 2️⃣ BM25 results
        bm25_docs = self.bm25_retriever.retrieve(query)

        # 3️⃣ Merge
        all_docs = vector_docs + bm25_docs

        # 4️⃣ Dedupe based on chunk_id or text
        seen = set()
        merged_docs = []
        for doc in all_docs:
            key = doc.metadata.get("chunk_id", doc.page_content[:30])
            if key not in seen:
                merged_docs.append(doc)
                seen.add(key)

        # 5️⃣ Return top_k
        return merged_docs[:top_k]
