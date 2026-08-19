import numpy as np
from voiceraghh.vectorstore import VectorStore

np.random.seed(42)
dimension = 128
texts = [f"Document {i} about topic {i % 5}" for i in range(20)]
embeddings = np.random.randn(20, dimension).astype(np.float32)

store = VectorStore(dimension=dimension)
store.add(texts, embeddings, [{"id": i} for i in range(20)])

query = np.random.randn(1, dimension).astype(np.float32)
results = store.search(query, k=3)

print(f"Query results:")
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

store.save("./data/vectorstore_test")
print("\nSaved!")
