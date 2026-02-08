import requests

class ReasoningAgent:
    def __init__(self, model="gemma3:4b", url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def answer(self, question, docs):
        context_blocks = []

        for i, doc in enumerate(docs):
            context_blocks.append(
                f"[Doc{i+1}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are an intelligent document-based QA assistant.

Answer the question ONLY using the documents below.
For every factual statement, cite the document ID like [Doc1], [Doc2].

Documents:
{context}

Question: {question}

Answer (with citations):
"""

        res = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return res.json()["response"]
