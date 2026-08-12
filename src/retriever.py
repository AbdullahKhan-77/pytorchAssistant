"""
retriever.py - Two-stage retrieval.
Stage 1: bi-encoder (BGE) fetches top-20 candidates from Chroma (wide net).
Stage 2: cross-encoder reranks those candidates by scoring each query-chunk pair
         jointly, and keeps the best few (sharper ranking than the bi-encoder alone).
Exposes retrieve(question) for use by generate.py / app.py.
Run:  python src/retriever.py   (self-test)
"""

import chromadb
from sentence_transformers import CrossEncoder

from embedder import embed_query

DB_DIR = "chroma_db"
COLLECTION_NAME = "pytorch_docs"
RETRIEVE_K = 20   # stage 1 candidates
FINAL_K = 5       # stage 2 survivors

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

print("Loading reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker loaded.")


def retrieve(question, retrieve_k=RETRIEVE_K, final_k=FINAL_K):
    """Return the top-`final_k` most relevant chunks for a question."""
    query_vec = embed_query(question)
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=retrieve_k,
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "source": results["metadatas"][0][i]["source"],
            "text": results["documents"][0][i],
        })

    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:final_k]


if __name__ == "__main__":
    hits = retrieve("How do I create a linear layer?")
    for rank, hit in enumerate(hits, start=1):
        print(f"--- Result {rank} (score {hit['rerank_score']:.3f}) ---")
        print("Source:", hit["source"])
        print(hit["text"][:200])
        print()