from voiceraghh.pipeline import VoiceRAG


documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "FastAPI is a modern web framework for building APIs with Python, based on standard Python type hints.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Speech recognition converts spoken language into text using machine learning models.",
    "Retrieval-Augmented Generation (RAG) combines search with language generation."
]

print("Building index...")
rag = VoiceRAG()
rag.build_index(documents)

queries = [
    "What is FAISS used for?",
    "How do you build APIs in Python?",
    "What is RAG?"
]

for q in queries:
    result = rag.answer(q)
    print(f"\nQ: {q}")
    print(f"A: {result.answer}")
    print(f"Sources: {len(result.sources)}")
