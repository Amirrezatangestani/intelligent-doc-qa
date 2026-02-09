from langgraph.graph import StateGraph, END
from src.agents.retriever import RetrieverAgent
from src.agents.reasoning import ReasoningAgent
from src.agents.utility import UtilityAgent
from src.retrieval.hybrid import HybridRetriever
from src.agents.state import QAState


class MultiAgentQA:
    def __init__(self, hybrid: HybridRetriever):
        self.retriever = RetrieverAgent(hybrid)
        self.reasoner = ReasoningAgent()
        self.utility = UtilityAgent()
        self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QAState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("reason", self._reason)
        graph.add_node("summarize", self._summarize)

        graph.set_entry_point("retrieve")

        # Conditional edge after retrieve
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "reason": "reason",
                "end": END,
            },
        )

        graph.add_edge("reason", "summarize")
        graph.add_edge("summarize", END)

        self.app = graph.compile()

    def _retrieve(self, state: QAState):
        docs = self.retriever.retrieve(state["question"])

        if not docs:
            return {
                "final_answer": "I couldn't find relevant information. Can you clarify your question?",
                "docs": []
            }

        return {"docs": docs}

    def _reason(self, state: QAState):
        reasoning = self.reasoner.answer(
            state["question"], state["docs"]
        )
        return {"reasoning_output": reasoning}

    def _summarize(self, state: QAState):
        ans = state["reasoning_output"]["answer"]
        used = state["reasoning_output"].get("used_chunks", [])

        docs = state.get("docs", [])

        citation_lines = []
        for i in used:
            if i < len(docs):
                meta = docs[i].metadata
                source = meta.get("source", "unknown")
                page = meta.get("page", "N/A")
                citation_lines.append(f"- {source} (page {page})")

        if citation_lines:
            citations = "\n\nSources:\n" + "\n".join(citation_lines)
        else:
            citations = ""

        return {"final_answer": ans + citations}

    def ask(self, question: str):
        return self.app.invoke({"question": question})


    def _route_after_retrieve(self, state: QAState):
        if "docs" not in state or not state["docs"]:
            return "end"
        return "reason"
