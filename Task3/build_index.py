from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

ROOT_DIR = Path(__file__).resolve().parent
KB_DIR = ROOT_DIR.parent / "knowledge_base"
CHROMA_DIR = ROOT_DIR.parent / "chroma_db"

def list_txt_files(kb_dir: Path):
    files = sorted(kb_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {kb_dir}")
    return files


def split_documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])
        for chunk_index, chunk in enumerate(text_chunks):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            chunks.append(
                {
                    "id": f"{doc['title']}::chunk_{chunk_index}",
                    "text": cleaned,
                    "source": doc["path"],
                    "title": doc["title"],
                    "chunk_index": chunk_index,
                    "total_chunks": len(text_chunks),
                }
            )
    return chunks


def build_vector_index(kb_dir=KB_DIR, chroma_dir=CHROMA_DIR):
    docs = []
    for path in list_txt_files(kb_dir):
        text = path.read_text(encoding="utf-8").strip()
        docs.append({
            "path": str(path),
            "title": path.name,
            "text": text,
        })

    chunks = split_documents_into_chunks(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(name="knowledge_base")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "title": chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }
        for chunk in chunks
    ]
    vectors = embeddings.embed_documents(texts)

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

def main():
    build_vector_index()
    print(f"Index created in: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
