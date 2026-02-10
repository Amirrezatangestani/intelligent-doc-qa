from langgraph.graph import StateGraph, END
from src.agents.state import QAState
from src.agents.intent import IntentAgent
from src.agents.retriever import RetrieverAgent
from src.agents.reasoning import ReasoningAgent
from src.agents.utility import UtilityAgent


class MultiAgentQA:
    def __init__(self, hybrid_retriever):
        self.intent_agent = IntentAgent()
        self.retriever = RetrieverAgent(hybrid_retriever)
        self.reasoner = ReasoningAgent()
        self.utility = UtilityAgent()
        self.memory = []
        self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QAState)

        graph.add_node("intent", self._intent)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("reason", self._reason)
        graph.add_node("utility", self._utility)

        graph.set_entry_point("intent")

        graph.add_conditional_edges(
            "intent",
            self._route_intent,
            {
                "answer": "retrieve",
                "summarize": "utility",
                "checklist": "utility",
                "translate": "utility",
            }
        )

        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "utility")
        graph.add_edge("utility", END)

        self.app = graph.compile()

    # -------- Nodes -------- #

    def _intent(self, state: QAState):
        result = self.intent_agent.classify(state["question"])
        return {
            "intent": result["intent"],
            "target_language": result.get("target_language")
        }

    def _route_intent(self, state: QAState):
        return state["intent"]

    def _retrieve(self, state: QAState):
        docs = self.retriever.retrieve(state["question"])
        print(f"Retrieved {len(docs)} documents")
        return {"docs": docs}

    def _reason(self, state: QAState):
        reasoning = self.reasoner.answer(
            state["question"],
            state["docs"],
            state.get("chat_history", [])
        )
        return {"reasoning_output": reasoning}

    def _utility(self, state: QAState):
        intent = state["intent"]

        if intent == "answer":
            return {"final_answer": state["reasoning_output"]["answer"]}

        if intent == "summarize":
            return {
                "final_answer": self.utility.summarize(state["question"])
            }

        if intent == "checklist":
            return {
                "final_answer": self.utility.checklist(state["question"])
            }

        if intent == "translate":
            return {
                "final_answer": self.utility.translate(
                    state["question"],
                    state["target_language"] or "English"
                )
            }

        return {"final_answer": "I don't know."}

    def ask(self, question: str):
        state = {
            "question": question,
            "chat_history": self.memory
        }

        result = self.app.invoke(state)

        # Save conversation turn
        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": result["final_answer"]})

        return result
