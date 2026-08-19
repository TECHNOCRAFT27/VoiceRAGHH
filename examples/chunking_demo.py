from voiceraghh.chunking import fixed_size_chunks, sentence_chunks, paragraph_chunks, recursive_chunks

sample_text = """This is a test document. It contains multiple sentences. We want to chunk this text in different ways.

This is the second paragraph. It also has sentences. The chunking strategies should handle this properly.

Here is a third paragraph with more content. The goal is to split text into meaningful chunks for retrieval."""

print("Fixed size chunks:")
for c in fixed_size_chunks(sample_text, chunk_size=100):
    print(f"  [{len(c.text)}] {c.text[:50]}...")

print("\nSentence chunks:")
for c in sentence_chunks(sample_text):
    print(f"  [{len(c.text)}] {c.text[:50]}...")

print("\nParagraph chunks:")
for c in paragraph_chunks(sample_text):
    print(f"  [{len(c.text)}] {c.text[:50]}...")

print("\nRecursive chunks:")
for c in recursive_chunks(sample_text, chunk_size=100):
    print(f"  [{len(c.text)}] {c.text[:50]}...")
