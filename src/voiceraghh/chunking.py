from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict


def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 0) -> list[Chunk]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, metadata={"strategy": "fixed_size", "start": start, "end": end}))
        start = end - overlap if overlap else end
    return chunks


def sentence_chunks(text: str) -> list[Chunk]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > 500 and current_chunk:
            chunks.append(Chunk(text=" ".join(current_chunk), metadata={"strategy": "sentence"}))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += len(sent)
    if current_chunk:
        chunks.append(Chunk(text=" ".join(current_chunk), metadata={"strategy": "sentence"}))
    return chunks


def paragraph_chunks(text: str) -> list[Chunk]:
    paragraphs = text.split("\n\n")
    return [Chunk(text=p.strip(), metadata={"strategy": "paragraph"}) for p in paragraphs if p.strip()]


def recursive_chunks(text: str, chunk_size: int = 500) -> list[Chunk]:
    if len(text) <= chunk_size:
        return [Chunk(text=text, metadata={"strategy": "recursive"})]
    
    for sep in ["\n\n", "\n", ". ", " "]:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                if len(current) + len(part) + len(sep) > chunk_size and current:
                    chunks.append(Chunk(text=current, metadata={"strategy": "recursive"}))
                    current = part
                else:
                    current = current + sep + part if current else part
            if current:
                chunks.append(Chunk(text=current, metadata={"strategy": "recursive"}))
            return chunks
    
    return [Chunk(text=text, metadata={"strategy": "recursive"})]
