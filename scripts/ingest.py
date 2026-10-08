"""Gera o índice vetorial (data/index/) a partir dos .md em data/raw/.

Uso: python scripts/ingest.py     (precisa de GEMINI_API_KEY no .env)
O índice é commitado no repositório, então o deploy não precisa recalculá-lo.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import llm  # noqa: E402
from app.chunking import Chunk, chunk_page  # noqa: E402
from app.config import INDEX_DIR, RAW_DIR  # noqa: E402

MIN_TEXT = 40


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> int:
    files = sorted(p for p in RAW_DIR.rglob("*.md") if " copy" not in p.name)
    if not files:
        print("Nenhum .md em data/raw/. Rode scripts/fetch_docs.py antes.")
        return 1

    chunks = []
    for p in files:
        rel = p.relative_to(RAW_DIR).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks += [c for c in chunk_page(rel, text) if len(c.text) >= MIN_TEXT]
    print(f"{len(files)} páginas -> {len(chunks)} trechos. Gerando embeddings...")

    # Reaproveita embeddings do índice anterior (mesmo texto = mesmo vetor): só o que mudou é recalculado.
    cache: dict[str, np.ndarray] = {}
    if (INDEX_DIR / "chunks.json").exists() and (INDEX_DIR / "vectors.npy").exists():
        old = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        old_vecs = np.load(INDEX_DIR / "vectors.npy").astype(np.float32)
        cache = {_key(Chunk(**c).embed_text()): v for c, v in zip(old, old_vecs)}

    # Progresso parcial: se a cota acabar no meio, o que já foi calculado não se perde.
    partial = RAW_DIR / ".embed_cache.npz"
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        cache.update(zip(z["keys"].tolist(), z["vecs"]))

    texts = [c.embed_text() for c in chunks]
    missing = sorted({t for t in texts if _key(t) not in cache})
    print(f"{len({_key(t) for t in texts}) - len(missing)} reaproveitados, {len(missing)} novos")
    for i in range(0, len(missing), llm.EMBED_BATCH):
        if i:
            print(f"  {i}/{len(missing)}; aguardando a cota por minuto (65s)...", flush=True)
            time.sleep(65)
        batch = missing[i : i + llm.EMBED_BATCH]
        for t, v in zip(batch, llm.embed(batch, task="RETRIEVAL_DOCUMENT")):
            cache[_key(t)] = v
        np.savez(partial, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    vectors = np.stack([cache[_key(t)] for t in texts])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    np.save(INDEX_DIR / "vectors.npy", vectors.astype(np.float16))
    print(f"Índice salvo em {INDEX_DIR} ({vectors.shape[0]} x {vectors.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
