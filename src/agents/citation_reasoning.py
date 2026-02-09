from typing import List
from langchain_core.documents import Document
from src.agents.reasoning import ReasoningAgent


class CitationReasoningAgent(ReasoningAgent):

    def answer_with_citations(self, question: str, docs: List[Document]) -> dict:
        answer = self.answer(question, docs)

        citations = []
        for doc in docs:
            citations.append({
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "preview": doc.page_content[:120]
            })

        return {
            "answer": answer,
            "citations": citations
        }