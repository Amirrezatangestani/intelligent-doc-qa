import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.llm.base import get_llm

class ReasoningAgent:
    def __init__(self):
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using ONLY the documents.\n"
             "If not found, answer: \"I don't know\".\n"
             "Return STRICT JSON."),
            ("human",
             "Question:\n{question}\n\n"
             "Documents:\n{docs}\n\n"
             "JSON format:\n"
             "{{"
             "\"answer\": \"...\", "
             "\"used_chunks\": [0, 1]"
             "}}")
        ])

    def answer(self, question: str, docs: List[Document]) -> Dict:
        formatted_docs = "\n".join(
            f"[{i}] {d.page_content[:800]}" for i, d in enumerate(docs)
        )

        response = self.llm.invoke(
            self.prompt.format_messages(
                question=question,
                docs=formatted_docs
            )
        ).content

        try:
            return json.loads(response)
        except Exception:
            return {
                "answer": response,
                "used_chunks": []
            }
