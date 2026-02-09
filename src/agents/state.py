from typing import TypedDict, List, Dict, Optional
from langchain_core.documents import Document


class QAState(TypedDict, total=False):
    # user input
    question: str

    # retriever output
    docs: List[Document]

    # reasoning agent output (structured)
    reasoning_output: Dict

    # final utility output
    final_answer: str