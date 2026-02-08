from langchain_community.retrievers import BM25Retriever


class BM25DocRetriever:
    def __init__(self, docs, k=3):
        self.retriever = BM25Retriever.from_documents(docs)
        self.retriever.k = k

    def retrieve(self, query: str):
        return self.retriever.invoke(query)
