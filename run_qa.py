from src.bootstrap import build_or_load_retriever
from src.agents.multi_agent import MultiAgentQA

hybrid = build_or_load_retriever()
qa = MultiAgentQA(hybrid)

print("\n🤖 Intelligent Document QA")
print("Drop files into the `docs/` directory.")
print("Type 'exit' to quit.\n")

while True:
    question = input("💬 Question: ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue

    result = qa.ask(question)
    print("\n🧠 Answer:\n")
    print(result["final_answer"])
    print("-" * 60)