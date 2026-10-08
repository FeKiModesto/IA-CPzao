import numpy as np

from app.retrieval import Index, tokenize

CHUNKS = [
    {"id": "a", "title": "MQTT", "section": "Broker", "url": "u/a", "text": "O broker MQTT recebe publicações dos clientes."},
    {"id": "b", "title": "RAG", "section": "Embeddings", "url": "u/b", "text": "Embeddings representam o significado de textos."},
    {"id": "c", "title": "ESP32", "section": "GPIO", "url": "u/c", "text": "Pinos digitais acendem um LED no ESP32."},
]
VECS = np.eye(3, dtype=np.float32)


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("A Publicação do Tópico!") == ["publicacao", "topico", "publicacaotopico"]


def test_tokenize_keeps_lab_numbers_and_joins_neighbours():
    assert "3.5" in tokenize("Lab 3.5") and "4" in tokenize("Lab 4")
    assert set(tokenize("Pet Shop")) & set(tokenize("petshop")) == {"petshop"}


def test_lexical_match_wins_when_semantic_is_neutral():
    idx = Index(CHUNKS, VECS)
    neutral = np.ones(3, dtype=np.float32) / np.sqrt(3)
    hits = idx.search("broker mqtt", neutral, k=2)
    assert hits[0].chunk["id"] == "a"


def test_semantic_match_wins_without_shared_words():
    idx = Index(CHUNKS, VECS)
    hits = idx.search("qualquer coisa sem palavra em comum", np.array([0, 1, 0], np.float32), k=1)
    assert hits[0].chunk["id"] == "b"
    assert hits[0].score == 1.0
