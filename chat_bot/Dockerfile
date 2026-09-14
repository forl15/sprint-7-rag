FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHROMA_DIR=/app/chroma_db \
    OLLAMA_BASE_URL=http://ollama:11434

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi uvicorn \
        chromadb \
        langchain \
        langchain-core \
        langchain-ollama \
        langchain-text-splitters \
        sentence_transformers

COPY knowledge_base ./knowledge_base
COPY chroma_db ./chroma_db
COPY chat_bot ./chat_bot

EXPOSE 8000

CMD ["python", "-m", "chat_bot.start_app"]
