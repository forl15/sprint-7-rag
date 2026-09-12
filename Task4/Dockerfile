FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_DIR=/app/chroma_db \
    OLLAMA_BASE_URL=http://ollama:11434

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi uvicorn \
        chromadb \
        langchain \
        langchain-core \
        langchain-ollama \
        langchain-text-splitters

COPY knowledge_base ./knowledge_base
COPY chroma_db ./chroma_db
COPY Task3 ./Task3
COPY Task4 ./Task4

EXPOSE 8000

CMD ["python", "/app/Task4/start_app.py"]
