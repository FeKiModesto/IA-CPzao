"""Índice de trechos com busca híbrida: embeddings (semântica) + BM25 (palavras exatas)."""
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import INDEX_DIR

_STOP = set(
    "a o as os um uma de do da dos das em no na nos nas para por com sem que e ou se ao aos "
    "como mais mas foi sao ser tem ter eu voce nos ele ela isso este esta esse essa qual quais "
    "quando onde sobre entre pelo pela the of and to in is".split()
)


def tokenize(text: str) -> list[str]:
    """Palavras sem acento/stopwords. Mantém números ("lab 3.5" -> "3.5") e acrescenta as junções
    de vizinhas ("pet shop" -> "petshop"), para casar grafias diferentes da mesma coisa."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    words = [
        t for t in re.findall(r"\d+(?:\.\d+)*|[a-z0-9_]+", text)
        if t not in _STOP and (len(t) > 1 or t.isdigit())
    ]
    return words + [a + b for a, b in zip(words, words[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


@dataclass
class Hit:
    chunk: dict
    score: float  # similaridade de cosseno com a pergunta
    rrf: float


class Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25([tokenize(f"{c['title']} {c['section']} {c['text']}") for c in chunks])

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "Index":
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy").astype(np.float32)
        return cls(chunks, vectors)

    def search(self, query: str, query_vec: np.ndarray | None, k: int = 6, pool: int = 25) -> list[Hit]:
        """Funde os rankings semântico e lexical por Reciprocal Rank Fusion (RRF).
        Sem query_vec (embedding indisponível) usa só o ranking lexical."""
        lex = self.bm25.scores(tokenize(query))
        if query_vec is None:
            cos = np.zeros(len(self.chunks), dtype=np.float32)
            semantic: list[int] = []
        else:
            cos = self.vectors @ query_vec
            semantic = [int(i) for i in np.argsort(-cos)[:pool]]
        lexical = [int(i) for i in np.argsort(-lex)[:pool] if lex[i] > 0]  # sem match = sem voto
        fused: dict[int, float] = {}
        for ranking in (semantic, lexical):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], float(cos[i]), fused[i]) for i in top]
