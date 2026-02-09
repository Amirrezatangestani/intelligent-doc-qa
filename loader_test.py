
from pprint import pprint
from pathlib import Path
from src.ingestion.loaders import load_document
from src.ingestion.indexer import load_vectorstore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from src.agents.citation_reasoning import CitationReasoningAgent
from pathlib import Path

# مسیر نمونه فایل‌ها


TEST_FILES = [
    "test_files/sample.txt",
    "test_files/sample.pdf",
    "test_files/sample.docx",
    "test_files/sample.png",
]



# یک سوال عمومی برای همه فایل‌ها
TEST_QUESTIONS = [
    {
        "question": "What is mentioned about information systems?",
        "keywords": ["information", "systems"]
    }
]

def run_loader_format_test():
    print("✅ Loading all test documents...")

    all_docs = []
    for f in TEST_FILES:
        try:
            docs = load_document(f)
            all_docs.extend(docs)
            print(f"[INFO] Loaded {len(docs)} docs from {f}")
        except Exception as e:
            print(f"[ERROR] Failed to load {f}: {e}")

    if not all_docs:
        print("[ERROR] No documents loaded. Exiting test.")
        return

    print("✅ Indexing loaded documents...")
    # فرض کنیم vectorstore قبلاً با docs ساخته شده و persist شده
    vectorstore = load_vectorstore()

    # Build retrievers
    bm25 = BM25DocRetriever(all_docs)
    vector = VectorRetriever(k=5)
    hybrid = HybridRetriever(vector, bm25)

    # Reasoning agent
    agent = CitationReasoningAgent()

    # اجرای QA روی تمام سوالات
    for idx, test in enumerate(TEST_QUESTIONS, start=1):
        question = test["question"]
        keywords = test["keywords"]

        print(f"\n🧪 Test {idx}: {question}")
        retrieved_docs = hybrid.retrieve(question)
        print(f"[INFO] Retrieved {len(retrieved_docs)} docs")

        result = agent.answer_with_citations(question, retrieved_docs)
        answer_text = result.get("answer", "")

        pprint(result)

        # بررسی keyword score
        hits = sum(1 for kw in keywords if kw.lower() in answer_text.lower())
        coverage = hits / len(keywords)
        if coverage >= 0.8:
            score = 1.0
            label = "✅ PASS"
        elif coverage >= 0.4:
            score = 0.5
            label = "⚠️ PARTIAL"
        else:
            score = 0.0
            label = "❌ FAIL"

        print(f"🎯 Keyword coverage: {coverage:.2f} => Score: {score} {label}")
        if score < 1.0:
            missing = [kw for kw in keywords if kw.lower() not in answer_text.lower()]
            print("⚠️ Missing keywords:", missing)

if __name__ == "__main__":
    run_loader_format_test()
