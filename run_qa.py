from src.ingestion.loaders import load_document
from src.ingestion.splitter import split_documents

from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25DocRetriever
from src.retrieval.hybrid import HybridRetriever

from src.agents.multi_agent import MultiAgentQA


# Load same docs
files = [
    "test_files/sample.pdf",
    "test_files/sample.txt",
    "test_files/sample.docx",
]

all_docs = []
for f in files:
    all_docs.extend(load_document(f))

chunks = split_documents(all_docs)

bm25 = BM25DocRetriever(chunks)
vector = VectorRetriever(k=5)
hybrid = HybridRetriever(vector, bm25)

qa = MultiAgentQA(hybrid)

question = "AI challenges?"

result = qa.ask(question)

print("\n🧠 FINAL ANSWER:")
print(result["final_answer"])
