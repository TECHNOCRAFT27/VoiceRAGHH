from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import os
import re
import hashlib
import time
import logging
from typing import Literal

from voiceraghh.stt import transcribe_file, transcribe_bytes
from voiceraghh.chunking import sentence_chunks, Chunk
from voiceraghh.vectorstore import VectorStore, SearchResult
from voiceraghh.embeddings import embed_texts, embed_query_cached


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceraghh")

UNSAFE_PATTERNS = [
    r"how to (make|build|create) (bomb|weapon|drug|explosive)",
    r"(kill|murder|harm) (someone|people|person)",
    r"(hack|crack|steal|fraud)",
    r"(suicide|self.harm)",
]

OFF_TOPIC_RESPONSE = "I can only answer questions based on the provided documents. Please ask a relevant question."

_response_cache: dict[str, str] = {}
MAX_RETRIES = 3
RETRY_DELAY = 0.5


class QueryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)
    language: str | None = None


class SourceResult(BaseModel):
    text: str
    score: float
    metadata: dict


class PipelineResult(BaseModel):
    query: str
    answer: str
    sources: list[SourceResult]
    confidence: float
    guardrail_triggered: bool = False
    guardrail_reason: str | None = None
    latency_ms: float = 0.0
    stage_timings: dict[str, float] = Field(default_factory=dict)
    status: Literal["success", "unsafe", "off_topic", "error"] = "success"


class VoiceRAG:
    def __init__(self, vectorstore_path: str | None = None):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vectorstore: VectorStore | None = None
        
        if vectorstore_path:
            self.vectorstore = VectorStore.load(vectorstore_path)

    def build_index(self, texts: list[str], metadata: list[dict] | None = None):
        chunks: list[Chunk] = []
        for text in texts:
            chunks.extend(sentence_chunks(text))
        
        chunk_texts = [c.text for c in chunks]
        
        import json
        cache_path = "./data/embeddings_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            embeddings_list = []
            for t in chunk_texts:
                if t in cache:
                    embeddings_list.append(cache[t])
                else:
                    emb = embed_texts([t])[0].tolist()
                    cache[t] = emb
                    embeddings_list.append(emb)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            embeddings = np.array(embeddings_list)
        else:
            embeddings = embed_texts(chunk_texts)
            cache = {t: e.tolist() for t, e in zip(chunk_texts, embeddings)}
            os.makedirs("./data", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        
        self.vectorstore = VectorStore(dimension=embeddings.shape[1])
        self.vectorstore.add(chunk_texts, embeddings, metadata or [c.metadata for c in chunks])

    def _is_unsafe(self, query: str) -> bool:
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in UNSAFE_PATTERNS)

    def _is_off_topic(self, query: str, sources: list[SearchResult], threshold: float = 0.3) -> bool:
        if not sources:
            return True
        avg_score = sum(s.score for s in sources) / len(sources)
        return avg_score < threshold

    def _call_llm_with_retry(self, messages: list[dict], **kwargs) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.groq_client.chat.completions.create(
                    model="allam-2-7b",
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 500)
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
        
        raise RuntimeError(f"LLM failed after {MAX_RETRIES} retries: {last_error}")

    def answer(self, input_data: QueryInput | str) -> PipelineResult:
        if isinstance(input_data, str):
            input_data = QueryInput(query=input_data)
        
        total_start = time.perf_counter()
        timings = {}
        
        if self._is_unsafe(input_data.query):
            return PipelineResult(
                query=input_data.query,
                answer="I cannot answer unsafe or harmful questions.",
                sources=[],
                confidence=0.0,
                guardrail_triggered=True,
                guardrail_reason="unsafe_content",
                status="unsafe",
                latency_ms=(time.perf_counter() - total_start) * 1000
            )
        
        embed_start = time.perf_counter()
        query_emb = embed_query_cached(input_data.query)
        timings["embed"] = (time.perf_counter() - embed_start) * 1000
        
        search_start = time.perf_counter()
        sources = self.vectorstore.search(query_emb, k=input_data.k)
        timings["search"] = (time.perf_counter() - search_start) * 1000
        
        if self._is_off_topic(input_data.query, sources):
            return PipelineResult(
                query=input_data.query,
                answer=OFF_TOPIC_RESPONSE,
                sources=[SourceResult(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
                confidence=0.0,
                guardrail_triggered=True,
                guardrail_reason="off_topic",
                status="off_topic",
                latency_ms=(time.perf_counter() - total_start) * 1000,
                stage_timings=timings
            )
        
        context = "\n\n".join(s.text for s in sources)
        
        cache_key = hashlib.md5((input_data.query + context).encode()).hexdigest()
        if cache_key in _response_cache:
            return PipelineResult(
                query=input_data.query,
                answer=_response_cache[cache_key],
                sources=[SourceResult(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
                confidence=sum(s.score for s in sources) / len(sources) if sources else 0,
                latency_ms=(time.perf_counter() - total_start) * 1000,
                stage_timings=timings
            )
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return PipelineResult(
                query=input_data.query,
                answer=f"[No API key] Retrieved context: {context[:200]}...",
                sources=[SourceResult(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
                confidence=sum(s.score for s in sources) / len(sources) if sources else 0,
                latency_ms=(time.perf_counter() - total_start) * 1000,
                stage_timings=timings,
                status="error"
            )
        
        llm_start = time.perf_counter()
        try:
            answer = self._call_llm_with_retry(
                messages=[
                    {"role": "system", "content": "Answer based on the context. If unsure, say you don't know."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {input_data.query}"}
                ]
            )
        except RuntimeError as e:
            return PipelineResult(
                query=input_data.query,
                answer=f"Error generating answer: {e}",
                sources=[SourceResult(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
                confidence=0.0,
                latency_ms=(time.perf_counter() - total_start) * 1000,
                stage_timings=timings,
                status="error"
            )
        timings["llm"] = (time.perf_counter() - llm_start) * 1000
        
        _response_cache[cache_key] = answer
        
        return PipelineResult(
            query=input_data.query,
            answer=answer,
            sources=[SourceResult(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
            confidence=sum(s.score for s in sources) / len(sources) if sources else 0,
            latency_ms=(time.perf_counter() - total_start) * 1000,
            stage_timings=timings
        )

    def voice_query(self, audio_path: str) -> PipelineResult:
        stt_start = time.perf_counter()
        result = transcribe_file(audio_path)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        
        pipeline_result = self.answer(QueryInput(query=result.text, language=result.language))
        pipeline_result.stage_timings["stt"] = stt_ms
        pipeline_result.query = f"[voice] {result.text}"
        return pipeline_result

    def voice_query_bytes(self, audio_bytes: bytes) -> PipelineResult:
        stt_start = time.perf_counter()
        result = transcribe_bytes(audio_bytes)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        
        pipeline_result = self.answer(QueryInput(query=result.text, language=result.language))
        pipeline_result.stage_timings["stt"] = stt_ms
        pipeline_result.query = f"[voice] {result.text}"
        return pipeline_result
