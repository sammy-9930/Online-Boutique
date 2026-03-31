from langchain_ollama import ChatOllama
from app.config import LLAMA_MODEL

def get_llama_llm():
    return ChatOllama(
        model=LLAMA_MODEL,
        temperature=0
    )