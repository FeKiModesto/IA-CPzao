import json

import numpy as np
import pytest

from app import llm, rag
from app.retrieval import Index

CHUNKS = [
    {"id": "a", "source": "a.md", "title": "MQTT", "section": "Broker", "url": "u/a",
     "text": "O broker MQTT recebe as publicações dos clientes e as entrega aos assinantes."},
    {"id": "b", "source": "b.md", "title": "RAG", "section": "Embeddings", "url": "u/b",
     "text": "Embeddings representam o significado de textos como listas de números."},
]


@pytest.fixture
def index():
    return Index(CHUNKS, np.eye(2, dtype=np.float32))


@pytest.fixture(autouse=True)
def clean_cache():
    rag._query_cache.clear()


def fake_generate(used=(1,), found=True):
    def gen(prompt, system, schema=None, temperature=0.2):
        return json.dumps({"found": found, "answer": "resposta [1]", "used_sources": list(used)}), "m"
    return gen


def test_answer_with_embeddings_returns_only_cited_sources(monkeypatch, index):
    monkeypatch.setattr(llm, "embed", lambda texts, task, patient=True: np.array([[0.0, 1.0]], np.float32))
    monkeypatch.setattr(llm, "generate", fake_generate(used=(1,)))
    res = rag.answer("o que são embeddings?", [], index)
    assert res.found and res.model == "m"
    assert [s["url"] for s in res.sources] == ["u/b"]  # trecho 1 = melhor ranqueado = RAG


def test_embedding_outage_falls_back_to_keyword_search(monkeypatch, index):
    def boom(*a, **k):
        raise RuntimeError("429 cota diária")
    monkeypatch.setattr(llm, "embed", boom)
    monkeypatch.setattr(llm, "generate", fake_generate(used=(1,)))
    res = rag.answer("como o broker MQTT funciona?", [], index)
    assert res.found and res.top_score is None
    assert res.sources[0]["url"] == "u/a" and res.sources[0]["score"] is None


def test_outage_with_no_keyword_match_says_not_found(monkeypatch, index):
    monkeypatch.setattr(llm, "embed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    called = []
    monkeypatch.setattr(llm, "generate", lambda *a, **k: called.append(1))
    res = rag.answer("receita de bolo", [], index)
    assert not res.found and res.answer == rag.NOT_FOUND and not called


def test_query_embedding_is_cached(monkeypatch, index):
    calls = []

    def emb(texts, task, patient=True):
        calls.append(texts)
        return np.array([[1.0, 0.0]], np.float32)

    monkeypatch.setattr(llm, "embed", emb)
    monkeypatch.setattr(llm, "generate", fake_generate())
    rag.answer("mesma pergunta", [], index)
    rag.answer("mesma pergunta", [], index)
    assert len(calls) == 1


def test_failed_embedding_is_not_cached(monkeypatch, index):
    monkeypatch.setattr(llm, "embed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert rag.embed_query("q") is None
    monkeypatch.setattr(llm, "embed", lambda *a, **k: np.array([[1.0, 0.0]], np.float32))
    assert rag.embed_query("q") is not None


def test_not_found_flag_hides_sources(monkeypatch, index):
    monkeypatch.setattr(llm, "embed", lambda *a, **k: np.array([[1.0, 0.0]], np.float32))
    monkeypatch.setattr(llm, "generate", fake_generate(used=(1, 2), found=False))
    res = rag.answer("algo fora do material", [], index)
    assert not res.found and res.sources == []
