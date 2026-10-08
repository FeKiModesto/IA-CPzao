import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, main, rag
from app.retrieval import Index


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    chunks = [{"id": "a", "title": "T", "section": "S", "url": "u", "text": "texto de teste"}]
    monkeypatch.setattr(Index, "load", classmethod(lambda cls, *a: cls(chunks, np.eye(1, dtype=np.float32))))
    monkeypatch.setattr(main, "_hits", main.defaultdict(main.deque))
    with TestClient(main.app) as c:
        yield c


def fake_answer(question, history, index):
    return rag.RagResult(
        answer=f"resp:{question}|hist={len(history)}", found=True,
        sources=[{"n": 1, "title": "T", "section": "S", "url": "u", "snippet": "x", "score": 0.9}],
        standalone_question=question, top_score=0.9,
    )


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["chunks"] == 1


def test_chat_persists_and_uses_history(client, monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer)
    r1 = client.post("/chat", json={"question": "primeira"}).json()
    sid = r1["session_id"]
    assert r1["answer"] == "resp:primeira|hist=0" and r1["sources"][0]["url"] == "u"

    r2 = client.post("/chat", json={"question": "segunda", "session_id": sid}).json()
    assert r2["answer"].endswith("hist=1")

    hist = client.get("/history", params={"session_id": sid}).json()
    assert [h["question"] for h in hist] == ["primeira", "segunda"]
    assert client.get("/stats").json()["interactions"] == 2


def test_history_survives_restart(tmp_path, monkeypatch):
    """Persistência: reabrir o app com o mesmo banco ainda mostra as conversas."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    chunks = [{"id": "a", "title": "T", "section": "S", "url": "u", "text": "t"}]
    monkeypatch.setattr(Index, "load", classmethod(lambda cls, *a: cls(chunks, np.eye(1, dtype=np.float32))))
    monkeypatch.setattr(rag, "answer", fake_answer)
    with TestClient(main.app) as c:
        sid = c.post("/chat", json={"question": "oi"}).json()["session_id"]
    with TestClient(main.app) as c:  # "reinício"
        assert len(c.get("/history", params={"session_id": sid}).json()) == 1


def test_feedback_only_for_own_session(client, monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer)
    r = client.post("/chat", json={"question": "q"}).json()
    ok = client.post("/feedback", json={"interaction_id": r["interaction_id"], "session_id": r["session_id"], "value": 1})
    assert ok.status_code == 200
    bad = client.post("/feedback", json={"interaction_id": r["interaction_id"], "session_id": "outra", "value": -1})
    assert bad.status_code == 404
    assert client.get("/stats").json()["thumbs_up"] == 1


def test_llm_failure_returns_502_and_logs_error(client, monkeypatch):
    def boom(*a):
        raise RuntimeError("api fora")
    monkeypatch.setattr(rag, "answer", boom)
    r = client.post("/chat", json={"question": "q"})
    assert r.status_code == 502
    assert client.get("/stats").json()["errors"] == 1


def test_exhausted_daily_quota_gets_a_clear_message(client, monkeypatch):
    from google.genai import errors

    def quota(*a):
        raise errors.ClientError(429, {"error": {"message": "quota"}})
    monkeypatch.setattr(rag, "answer", quota)
    r = client.post("/chat", json={"question": "q"})
    assert r.status_code == 503 and "cota" in r.json()["detail"]


def test_validation_and_rate_limit(client, monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer)
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 2)
    codes = [client.post("/chat", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
