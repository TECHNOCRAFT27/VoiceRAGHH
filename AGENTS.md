# AGENTS.md

## Project

Voice RAG system: speech-to-text → retrieval → answer generation. Uses HuggingFace dataset `ai4bharat/MSMARCO-XI`.

## Setup

- **Python 3.14** required (see `.python-version`)
- **uv** for dependency management: `uv sync`
- Environment vars in `.env`: `GROQ_API_KEY`, `WHISPER_MODEL_SIZE`, `FAISS_INDEX_PATH`

## Key Files

- `main.py` — standalone script to fetch sample data from HuggingFace (not the app entrypoint)
- `src/voiceraghh/__init__.py` — package placeholder (not yet implemented)
- `task 2_ hhg.md` — full task requirements

## Commands

```bash
uv sync                    # install dependencies
uv run python main.py      # fetch sample data
```

No lint, typecheck, or test commands configured yet.
