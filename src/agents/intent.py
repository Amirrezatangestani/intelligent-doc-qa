import json
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.llm.base import get_llm


class IntentAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an intent classifier for a document QA system.\n"
             "Decide the user's intent.\n\n"
             "Possible intents:\n"
             "- answer (question answering)\n"
             "- summarize (summary request)\n"
             "- checklist (step-by-step or bullet list)\n"
             "- translate (language translation)\n\n"
             "Return STRICT JSON only."),
            ("human",
             "User input:\n{question}\n\n"
             "JSON format:\n"
             "{{"
             "\"intent\": \"answer|summarize|checklist|translate\", "
             "\"target_language\": \"<language or null>\""
             "}}")
        ])

    def classify(self, question: str) -> Dict:
        response = self.llm.invoke(
            self.prompt.format_messages(question=question)
        ).content

        try:
            return json.loads(response)
        except Exception:
            return {
                "intent": "answer",
                "target_language": None
            }
