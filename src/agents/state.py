from typing import TypedDict, List, Dict, Optional
from langchain_core.documents import Document


class QAState(TypedDict, total=False):
    question: str

    # intent routing
    intent: str
    target_language: Optional[str]
    chat_history: List[Dict]
    # retrieval
    docs: List[Document]

    # reasoning
    reasoning_output: Dict

    # final
    final_answer: str