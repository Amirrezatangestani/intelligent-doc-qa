import json
from pprint import pprint

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from src.agents.citation_reasoning import CitationReasoningAgent
from src.ingestion.indexer import load_vectorstore


def load_tests(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_score(answer: str, keywords: list[str]) -> float:
    answer = answer.lower()

    hits = sum(1 for kw in keywords if kw.lower() in answer)
    coverage = hits / len(keywords) if keywords else 0.0

    if coverage >= 0.8:
        score = 1.0
    elif coverage >= 0.4:
        score = 0.5
    else:
        score = 0.0

    # penalty: very short / vague answers
    if len(answer.split()) < 8:
        score = max(score - 0.2, 0.0)

    return score


def run_qa_tests(tests):

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search("test", k=10)
    bm25 = BM25DocRetriever(docs)
    vector = VectorRetriever(k=5)
    hybrid = HybridRetriever(vector, bm25)

    agent = CitationReasoningAgent()

    print("🧪 Running QA Evaluation...\n")

    scores = []

    for idx, test in enumerate(tests, start=1):
        question = test["question"]
        keywords = test["keywords"]

        print(f"Test {idx}: {question}")
        print("🔍 Retrieving documents...")

        retrieved_docs = hybrid.retrieve(question)

        print(f"[INFO] Retrieved {len(retrieved_docs)} docs")
        for i, d in enumerate(retrieved_docs):
            preview = d.page_content[:200].replace("\n", " ")
            print(f"  [{i}] {preview} ...")

        result = agent.answer_with_citations(question, retrieved_docs)

        print("\n[RESULT]")
        pprint(result)

        answer_text = result.get("answer", "")
        score = keyword_score(answer_text, keywords)
        scores.append(score)

        # ---- grading ----
        if score == 1.0:
            label = "✅ PASS"
        elif score >= 0.5:
            label = "⚠️ PARTIAL"
        else:
            label = "❌ FAIL"

        print(f"\n🎯 Keyword score: {score:.2f}")
        print("Expected keywords:", keywords)
        print("Result:", label)

        if score < 1.0:
            missing = [kw for kw in keywords if kw.lower() not in answer_text.lower()]
            print("⚠️ Missing concepts:", missing)

        print("-" * 60)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"🧮 Average QA score: {avg_score:.2f}")


if __name__ == "__main__":
    tests = load_tests("qa_tests.json")
    run_qa_tests(tests)
