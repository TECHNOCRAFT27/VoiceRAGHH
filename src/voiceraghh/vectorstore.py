import faiss
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None):
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        distances, indices = self.index.search(query_embedding.astype(np.float32), min(k, len(self.texts)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append(SearchResult(
                    text=self.texts[idx],
                    score=float(1 / (1 + dist)),
                    metadata=self.metadata[idx]
                ))
        return results

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        import json
        with open(p / "texts.json", "w") as f:
            json.dump({"texts": self.texts, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        import json
        p = Path(path)
        index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "texts.json") as f:
            data = json.load(f)
        store = cls(dimension=index.d)
        store.index = index
        store.texts = data["texts"]
        store.metadata = data["metadata"]
        return store
