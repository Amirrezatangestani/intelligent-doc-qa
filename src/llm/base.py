from langchain_ollama import ChatOllama
import os

def get_llm(
    model: str = "gemma3:4b",
    temperature: float = 0.1,
    num_ctx: int = 4096,
):
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    )
