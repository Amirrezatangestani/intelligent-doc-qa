from typing import List
from langchain_core.documents import Document
from src.agents.reasoning import ReasoningAgent


class CitationReasoningAgent(ReasoningAgent):

    def answer_with_citations(self, question: str, docs: List[Document]) -> dict:
        result = self.answer(question, docs)

        used_indices = result.get("used_chunks", [])
        citations = []

        for i, idx in enumerate(used_indices, start=1):
            if idx >= len(docs):
                continue

            doc = docs[idx]

            citations.append({
                "id": f"[{i}]",
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "n/a"),
                "chunk_id": doc.metadata.get("chunk_id", f"chunk_{idx}"),
                "preview": doc.page_content.strip()[:120]
            })

        return {
            "answer": result["answer"],
            "citations": citations
        }
