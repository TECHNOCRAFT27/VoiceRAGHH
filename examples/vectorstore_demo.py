from voiceraghh.embeddings import embed_texts, embed_query
from voiceraghh.vectorstore import VectorStore

texts = [
    "Python is a programming language.",
    "Machine learning uses algorithms to learn from data.",
    "FastAPI is a web framework for building APIs.",
    "FAISS is a library for vector similarity search.",
    "Speech recognition converts audio to text."
]

print("Embedding texts...")
embeddings = embed_texts(texts)

store = VectorStore(dimension=embeddings.shape[1])
store.add(texts, embeddings, [{"text": t} for t in texts])

query = "What is used for vector search?"
query_emb = embed_query(query)

results = store.search(query_emb, k=2)
print(f"\nQuery: {query}")
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

store.save("./data/vectorstore")
print("\nSaved to ./data/vectorstore")
