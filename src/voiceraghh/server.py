from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from voiceraghh.pipeline import VoiceRAG, QueryInput, PipelineResult

load_dotenv()

app = FastAPI(title="Voice RAG API")
rag: VoiceRAG | None = None

static_dir = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    latency_ms: float
    guardrail_triggered: bool = False
    guardrail_reason: str | None = None
    stage_timings: dict[str, float] = {}
    status: str = "success"


@app.on_event("startup")
async def startup():
    global rag
    index_path = Path("./data/index")
    
    if index_path.exists() and (index_path / "index.faiss").exists():
        rag = VoiceRAG(vectorstore_path=str(index_path))
    else:
        import logging
        logger = logging.getLogger("voiceraghh")
        logger.warning("Index not found at ./data/index. Queries will fail.")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag:
        raise HTTPException(503, "Index not loaded")
    
    result: PipelineResult = rag.answer(QueryInput(query=req.query, k=req.k))
    
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[{"text": s.text[:100], "score": s.score} for s in result.sources],
        latency_ms=result.latency_ms,
        guardrail_triggered=result.guardrail_triggered,
        guardrail_reason=result.guardrail_reason,
        stage_timings=result.stage_timings,
        status=result.status
    )


@app.post("/voice", response_model=QueryResponse)
async def voice(file: UploadFile = File(...), k: int = 3):
    if not rag:
        raise HTTPException(503, "Index not loaded")
    
    audio_bytes = await file.read()
    result: PipelineResult = rag.voice_query_bytes(audio_bytes)
    
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[{"text": s.text[:100], "score": s.score} for s in result.sources],
        latency_ms=result.latency_ms,
        guardrail_triggered=result.guardrail_triggered,
        guardrail_reason=result.guardrail_reason,
        stage_timings=result.stage_timings,
        status=result.status
    )


@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": rag is not None}


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(static_dir / "index.html")


app.mount("/static", StaticFiles(directory=static_dir), name="static")
