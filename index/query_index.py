from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = ROOT_DIR.parent / "chroma_db"


def query_vector_index(query: str, top_k: int = 5, chroma_dir=CHROMA_DIR):

    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name="knowledge_base")

    query_embedding = embedding_model.encode_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "rank": i + 1,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return formatted


def main():
    query = "Чем известен Фарад Орион?"
    results = query_vector_index(query, top_k=5)
    print(f"Query: {query}\n")
    for result in results:
        print(f"Rank {result['rank']} | distance={result['distance']:.4f} | source={result['metadata']['source']}")
        print(result['text'][:800])
        print("-" * 80)


if __name__ == "__main__":
    main()
