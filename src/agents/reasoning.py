import json
import re
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm.base import get_llm


class ReasoningAgent:
    def __init__(self):
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a reasoning agent for a Retrieval-Augmented Generation system.\n"
                "You MUST answer using ONLY the provided documents.\n"
                "Conversation history is provided ONLY for resolving references.\n"
                "Do NOT use outside knowledge.\n"
                "If the answer is not explicitly found in the documents, respond with:\n"
                "\"I don't know\".\n\n"
                "IMPORTANT:\n"
                "- Return STRICT JSON only\n"
                "- Do NOT include markdown\n"
                "- Do NOT include explanations outside JSON\n"
                "- The JSON must contain exactly two keys: 'answer' and 'used_chunks'\n"
            ),
            (
                "human",
                "Conversation History:\n{history}\n\n"
                "Question:\n{question}\n\n"
                "Documents:\n{docs}\n\n"
                "Return JSON in this exact format:\n"
                "{{\n"
                "  \"answer\": \"...\",\n"
                "  \"used_chunks\": [0, 1]\n"
                "}}"
            )
        ])

    @staticmethod
    def _clean_json(text: str) -> str:
        """
        Removes markdown code fences and extra text
        produced by Gemma / Ollama models.
        """
        text = text.strip()

        # Remove ```json or ``` wrappers
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)

        return text.strip()

    def answer(
            self,
            question: str,
            docs: List[Document],
            history: List[Dict]
    ) -> Dict:

        formatted_docs = "\n".join(
            f"[{i}] {doc.page_content[:800]}"
            for i, doc in enumerate(docs)
        )
        print("---- DOCS SENT TO LLM ----")
        print(formatted_docs[:500])
        print("--------------------------")

        formatted_history = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in history
        )

        raw_response = self.llm.invoke(
            self.prompt.format_messages(
                question=question,
                docs=formatted_docs,
                history=formatted_history
            )
        ).content

        try:
            clean = self._clean_json(raw_response)
            parsed = json.loads(clean)

            if "answer" not in parsed or "used_chunks" not in parsed:
                raise ValueError("Invalid JSON schema")

            return parsed

        except Exception:
            return {
                "answer": raw_response.strip(),
                "used_chunks": []
            }

