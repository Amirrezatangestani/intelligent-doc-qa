from langchain_core.prompts import ChatPromptTemplate

from src.llm.base import get_llm


class UtilityAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def summarize(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize clearly in 3-5 sentences."),
            ("human", "{text}")
        ])
        return self.llm.invoke(
            prompt.format_messages(text=text)
        ).content.strip()

    def checklist(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Convert into a bullet-point checklist."),
            ("human", "{text}")
        ])
        return self.llm.invoke(
            prompt.format_messages(text=text)
        ).content.strip()

    def translate(self, text: str, target_lang: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the text into {target_lang}."),
            ("human", "{text}")
        ])
        return self.llm.invoke(
            prompt.format_messages(text=text)
        ).content.strip()
