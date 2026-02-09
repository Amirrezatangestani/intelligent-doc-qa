from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.hybrid import HybridRetriever
from src.llm.base import get_llm


class RetrieverAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the question into a short keyword search query. "
             "Do not explain."),
            ("human", "{question}")
        ])

    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        print("🔍 RetrieverAgent: optimizing query")

        query = self.llm.invoke(
            self.prompt.format_messages(question=question)
        ).content.strip()

        print("🔍 RetrieverAgent: retrieving documents")
        return self.retriever.retrieve(query, top_k=top_k)