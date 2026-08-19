import numpy as np
from fastembed import TextEmbedding


_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = list(model.embed(texts))
    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])


_cache: dict[str, np.ndarray] = {}


def embed_query_cached(query: str) -> np.ndarray:
    if query not in _cache:
        _cache[query] = embed_query(query)
    return _cache[query]


def clear_cache():
    _cache.clear()
