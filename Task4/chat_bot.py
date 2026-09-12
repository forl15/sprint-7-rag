import os
from pathlib import Path

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_db")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="Chat Bot")


class QueryRequest(BaseModel):
    query: str


if not CHROMA_DIR.exists():
    raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DIR}. Run Task3/build_index.py first.")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name="knowledge_base")

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url=OLLAMA_BASE_URL,
)

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.1,
    base_url=OLLAMA_BASE_URL,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — полезный ассистент. Отвечай строго по предоставленному контексту. "
            "Если в контексте нет ответа, скажи, что информации недостаточно. "
            "Не придумывай факты и указывай источники, когда это уместно.",
        ),
        (
            "user",
            "Контекст:\n{context}\n\nВопрос:\n{question}",
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


@app.post("/query")
def query_rag(payload: QueryRequest):
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        query_embedding = embedding_model.embed_query(query_text)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector search is unavailable: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return {"answer": "В базе знаний нет релевантных фрагментов для этого вопроса.", "sources": []}

    context_parts = []
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown") if metadata else "unknown"
        title = metadata.get("title", "unknown") if metadata else "unknown"
        context_parts.append(f"[Источник: {source} | Файл: {title}]\n{document}")

    context = "\n\n".join(context_parts)
    try:
        answer = chain.invoke({"context": context, "question": query_text})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM is unavailable: {exc}") from exc

    return {
        "answer": answer,
        "sources": [
            {
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", "unknown"),
                "chunk_index": metadata.get("chunk_index"),
            }
            for metadata in metadatas
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
