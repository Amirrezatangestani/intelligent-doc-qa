from langgraph import Graph, Node  
from src.retrieval.hybrid import HybridRetriever
from src.agents.reasoning import ReasoningAgent

class MultiAgentQA:
    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid
        self.reasoning_agent = ReasoningAgent()
        self.graph = Graph()
        self._setup_graph()

    def _setup_graph(self):
        # Node برای پاسخ به سوال
        self.qa_node = Node(func=self._reasoning_qa)
        self.graph.add_node(self.qa_node)

        # Node برای خلاصه کردن
        self.summary_node = Node(func=self._reasoning_summary)
        self.graph.add_node(self.summary_node)

        # می‌تونی ارتباط بین Node ها رو هم تعریف کنی
        # self.graph.add_edge(self.qa_node, self.summary_node)

    def _reasoning_qa(self, question, docs):
        return self.reasoning_agent.answer(question, docs)

    def _reasoning_summary(self, question, docs):
        answer = self.reasoning_agent.answer(question, docs)
        # مثال ساده خلاصه
        return answer[:300] + "..." if len(answer) > 300 else answer

    def ask(self, question):
        docs = self.hybrid.retrieve(question, top_k=5)
        # اجرای Node QA
        return self.graph.run(self.qa_node, question, docs)
