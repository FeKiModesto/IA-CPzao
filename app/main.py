import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from google.genai import errors
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import config, db, rag
from .retrieval import Index

log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


class State:
    index: Index | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    try:
        State.index = Index.load()
        log.info("Índice carregado: %d trechos", len(State.index.chunks))
    except FileNotFoundError:
        log.error("Índice não encontrado em data/index. Rode scripts/ingest.py.")
    yield


app = FastAPI(title="Assistente Disruptive Architectures", version="1.0.0", lifespan=lifespan)


# --- rate limit simples por IP (em memória; suficiente para uma instância) ---
_hits: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(request: Request) -> None:
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    now, window = time.monotonic(), _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Muitas perguntas em pouco tempo. Aguarde um minuto.")
    window.append(now)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)


class Source(BaseModel):
    n: int
    title: str
    section: str
    url: str
    snippet: str
    score: float | None = None


class ChatResponse(BaseModel):
    interaction_id: int
    session_id: str
    answer: str
    found: bool
    sources: list[Source]
    latency_ms: int


class FeedbackRequest(BaseModel):
    interaction_id: int
    session_id: str
    value: int = Field(description="1 = útil, -1 = não ajudou")


@app.get("/health")
def health():
    return {
        "status": "ok" if State.index else "sem_indice",
        "chunks": len(State.index.chunks) if State.index else 0,
        "model": config.GEMINI_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _check_rate_limit(request)
    if State.index is None:
        raise HTTPException(503, "Índice de documentos não carregado.")

    question = req.question.strip()
    session_id = req.session_id or uuid.uuid4().hex
    history = [(t.question, t.answer) for t in db.recent_turns(session_id)]

    start = time.perf_counter()
    try:
        result = rag.answer(question, history, State.index)
    except Exception as e:  # falha de API/rede: registra e devolve erro amigável
        log.exception("Falha ao responder")
        db.save_interaction(
            session_id=session_id, question=question, answer="", found=False, sources=[],
            model=config.GEMINI_MODEL, latency_ms=int((time.perf_counter() - start) * 1000),
            error=f"{type(e).__name__}: {e}"[:2000],
        )
        if isinstance(e, errors.APIError) and e.code == 429:
            raise HTTPException(
                503, "A cota gratuita da API do Gemini acabou por hoje. Tente novamente mais tarde."
            ) from e
        raise HTTPException(502, "Não consegui consultar o modelo agora. Tente novamente.") from e

    latency = int((time.perf_counter() - start) * 1000)
    row = db.save_interaction(
        session_id=session_id, question=question, standalone_question=result.standalone_question,
        answer=result.answer, found=result.found, sources=result.sources,
        top_score=result.top_score, model=result.model or config.GEMINI_MODEL, latency_ms=latency,
    )
    return ChatResponse(
        interaction_id=row.id, session_id=session_id, answer=result.answer,
        found=result.found, sources=result.sources, latency_ms=latency,
    )


@app.get("/history")
def history(session_id: str, limit: int = 50):
    with db.session() as s:
        rows = s.scalars(
            select(db.Interaction)
            .where(db.Interaction.session_id == session_id, db.Interaction.error.is_(None))
            .order_by(db.Interaction.id)
            .limit(min(limit, 200))
        ).all()
    return [
        {"interaction_id": r.id, "question": r.question, "answer": r.answer, "found": r.found,
         "sources": r.sources, "feedback": r.feedback, "created_at": r.created_at}
        for r in rows
    ]


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.value not in (1, -1):
        raise HTTPException(422, "value deve ser 1 ou -1")
    if not db.set_feedback(req.interaction_id, req.session_id, req.value):
        raise HTTPException(404, "Interação não encontrada.")
    return {"ok": True}


@app.get("/stats")
def stats():
    with db.session() as s:
        I = db.Interaction
        total = s.scalar(select(func.count(I.id))) or 0
        errors = s.scalar(select(func.count(I.id)).where(I.error.is_not(None))) or 0
        not_found = s.scalar(select(func.count(I.id)).where(I.found.is_(False), I.error.is_(None))) or 0
        up = s.scalar(select(func.count(I.id)).where(I.feedback == 1)) or 0
        down = s.scalar(select(func.count(I.id)).where(I.feedback == -1)) or 0
        avg = s.scalar(select(func.avg(I.latency_ms)).where(I.error.is_(None)))
    return {"interactions": total, "errors": errors, "not_found": not_found,
            "thumbs_up": up, "thumbs_down": down, "avg_latency_ms": int(avg) if avg else None}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
