import time
import statistics
from voiceraghh.pipeline import VoiceRAG
from voiceraghh.embeddings import clear_cache
import os
import json


documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "FastAPI is a modern web framework for building APIs with Python, based on standard Python type hints.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Speech recognition converts spoken language into text using machine learning models.",
    "Retrieval-Augmented Generation (RAG) combines search with language generation."
]

queries = [
    "What is FAISS?",
    "How to build APIs?",
    "What is RAG?",
    "Tell me about Python",
    "How does speech recognition work?"
]

print("Building index...")
rag = VoiceRAG()
rag.build_index(documents)

print("\n=== First run (uncached) ===")
clear_cache()
latencies_first = []
for q in queries:
    start = time.perf_counter()
    result = rag.answer(q)
    latency = (time.perf_counter() - start) * 1000
    latencies_first.append(latency)

print(f"P50:  {statistics.median(latencies_first):.1f}ms")
print(f"P100: {max(latencies_first):.1f}ms")

print("\n=== Second run (cached) ===")
latencies_cached = []
for q in queries:
    start = time.perf_counter()
    result = rag.answer(q)
    latency = (time.perf_counter() - start) * 1000
    latencies_cached.append(latency)

print(f"P50:  {statistics.median(latencies_cached):.1f}ms")
print(f"P100: {max(latencies_cached):.1f}ms")

print("\n=== Summary ===")
print(f"First run mean: {statistics.mean(latencies_first):.1f}ms")
print(f"Cached mean:    {statistics.mean(latencies_cached):.1f}ms")
print(f"Speedup:        {statistics.mean(latencies_first) / statistics.mean(latencies_cached):.1f}x")
