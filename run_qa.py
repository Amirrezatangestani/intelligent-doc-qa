from src.ingestion.loaders import load_document
from src.ingestion.splitter import split_documents

from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from src.retrieval.hybrid import HybridRetriever

from src.agents.multi_agent import MultiAgentQA

# -------------------- LOAD DOCUMENTS -------------------- #
files = [
    "test_files/sample.pdf",
    "test_files/sample.txt",
    "test_files/sample.docx",
]

all_docs = []
for f in files:
    all_docs.extend(load_document(f))

chunks = split_documents(all_docs)

# -------------------- BUILD RETRIEVER -------------------- #
bm25 = BM25DocRetriever(chunks)
vector = VectorRetriever(k=5)
hybrid = HybridRetriever(vector, bm25)

qa = MultiAgentQA(hybrid)

# -------------------- INTERACTIVE CLI -------------------- #
print("\n🤖 Welcome to Intelligent Document QA CLI!")
print("Type your questions below. Type 'exit' to quit.\n")

while True:
    question = input("💬 Your question: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break
    if not question:
        continue

    try:
        result = qa.ask(question)
    except Exception as e:
        print(f"[ERROR] Failed to get answer: {e}")
        continue

    answer = result.get("final_answer", "")
    sources = result.get("sources", [])

    if not answer.strip():
        print("⚠️ Sorry, no answer found for your query.\n")
        continue

    print("\n🧠 Answer:")
    print(answer)
    if sources:
        print("\n📚 Sources:")
        for s in sources:
            # source = filename, page = n/a, preview first 120 chars
            src_name = s.get("source", "unknown")
            page = s.get("page", "N/A")
            preview = s.get("preview", s.get("page_content", ""))[:120]
            print(f"- {src_name} (page {page}): {preview} ...")
    print("-" * 50)
