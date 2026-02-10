import pickle
from pathlib import Path
from langchain_community.retrievers import BM25Retriever


class BM25DocRetriever:
    def __init__(self, docs=None, k=3):
        self.k = k
        self.retriever = None

        if docs is not None:
            self.retriever = BM25Retriever.from_documents(docs)
            self.retriever.k = k

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.retriever, f)

    @classmethod
    def load(cls, path: str, k=3):
        with open(path, "rb") as f:
            retriever = pickle.load(f)
        obj = cls(k=k)
        obj.retriever = retriever
        obj.retriever.k = k
        return obj

    def retrieve(self, query: str):
        return self.retriever.invoke(query)