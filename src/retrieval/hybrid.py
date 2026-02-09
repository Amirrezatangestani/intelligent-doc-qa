from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever


class HybridRetriever:
    """
    Combine Vector Retriever and BM25 Retriever results
    and rerank them using a local LLM (Ollama - Gemma).
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25DocRetriever,
        llm=None,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.llm = llm or ChatOllama(model="gemma3:4b", temperature=0)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # 1️⃣ Vector results
        vector_docs = self.vector_retriever.retrieve(query)

        # 2️⃣ BM25 results
        bm25_docs = self.bm25_retriever.retrieve(query)

        # 3️⃣ Merge
        all_docs = vector_docs + bm25_docs

        # 4️⃣ Dedupe
        seen = set()
        merged_docs = []
        for doc in all_docs:
            key = doc.metadata.get("chunk_id", doc.page_content[:30])
            if key not in seen:
                merged_docs.append(doc)
                seen.add(key)

        # 5️⃣ Rerank with Gemma
        reranked = self._rerank_with_llm(query, merged_docs)

        return reranked[:top_k]

    def _rerank_with_llm(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        prompt = f"""
You are ranking document chunks by relevance to a user question.

Question:
{query}

Chunks:
"""

        for i, doc in enumerate(docs):
            prompt += f"\n[{i}] {doc.page_content[:400]}\n"

        prompt += """
Return only the indices sorted by most relevant to least relevant.
Format: comma-separated numbers.
Example: 2,0,1
"""

        response = self.llm.invoke(prompt).content.strip()

        try:
            order = [int(i) for i in response.split(",")]
            reranked_docs = [docs[i] for i in order if i < len(docs)]
            return reranked_docs
        except:
            # fallback if model returns garbage
            return docs
