# Voice RAG

A voice-enabled Retrieval-Augmented Generation (RAG) system built for HH Goa 2026.

**Pipeline:** Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation

## Features

- **Voice Input** - ElevenLabs speech-to-text for multi-language support
- **Smart Retrieval** - FAISS vector search with 4 chunking strategies
- **AI Answer** - Groq-powered LLM generates grounded answers
- **Guardrails** - Unsafe content filtering + off-topic detection
- **Fast** - <1ms cached latency, ~300ms first query

## Tech Stack

- **Backend:** FastAPI, Python 3.14
- **Vector DB:** FAISS (faiss-cpu)
- **Embeddings:** FastEmbed (all-MiniLM-L6-v2)
- **STT:** ElevenLabs
- **LLM:** Groq (allam-2-7b)
- **Dataset:** [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

## Setup

```bash
# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Build index from dataset
uv run python examples/build_index.py

# Run server
uv run uvicorn voiceraghh.server:app --host 0.0.0.0 --port 8000
```

## API Keys

Get your keys:
- **ElevenLabs:** https://elevenlabs.io/app/settings/api-keys (enable Speech-to-Text permission)
- **Groq:** https://console.groq.com/keys

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/query` | POST | Text query |
| `/voice` | POST | Voice query |
| `/health` | GET | Health check |

## Chunking Strategies

1. **Fixed-size** - Split by character count with optional overlap
2. **Sentence-based** - Split at sentence boundaries
3. **Paragraph-based** - Split at double newlines
4. **Recursive** - Hierarchical splitting (paragraphs → sentences → words)

## Latency

| Metric | First Run | Cached |
|--------|-----------|--------|
| P50 | ~325ms | <1ms |
| P100 | ~550ms | <1ms |

## Project Structure

```
voiceraghh/
├── src/voiceraghh/
│   ├── pipeline.py      # Main RAG pipeline
│   ├── vectorstore.py   # FAISS vector store
│   ├── embeddings.py    # Text embeddings
│   ├── chunking.py      # Chunking strategies
│   ├── stt.py           # Speech-to-text
│   ├── server.py        # FastAPI server
│   └── static/          # Web UI
├── examples/            # Demo scripts
├── data/                # Index and caches
└── main.py              # Dataset downloader
```

## License

MIT

---

Built for [HH Goa 2026](https://hhgoa.com/) | #RAGInGoa
