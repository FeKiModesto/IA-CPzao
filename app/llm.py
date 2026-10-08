"""Chamadas ao Gemini (embeddings e geração) com retry simples para limites da API."""
import random
import time

import httpx
import numpy as np
from google import genai
from google.genai import errors, types

from . import config

try:  # redes com inspeção de HTTPS (ex.: laboratórios): usa os certificados do sistema
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # em produção (Docker) não é necessário
    pass

EMBED_BATCH = 90
_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY não configurada (veja .env.example).")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _retry(fn, attempts: int = 8, wait_429: float = 45):
    """Repete em erros transitórios. Na ingestão a cota por minuto pede ~45s de espera (wait_429);
    em requisições de usuário usamos esperas curtas e trocamos de modelo (ver generate)."""
    for i in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 503, 504) or i == attempts - 1:
                raise
            time.sleep(wait_429 if code == 429 else min(2**i * 2, 30) + random.random())
        except httpx.TransportError:  # queda de conexão/SSL momentânea
            if i == attempts - 1:
                raise
            time.sleep(min(2**i, 10) + random.random())


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


def embed(texts: list[str], task: str, patient: bool = True) -> np.ndarray:
    """task: RETRIEVAL_DOCUMENT (trechos) ou RETRIEVAL_QUERY (perguntas). Retorna vetores unitários.
    patient=True (ingestão) espera a cota por minuto; False (usuário na fila) falha rápido."""
    attempts, wait_429 = (8, 45) if patient else (2, 1)
    vecs = []
    # A cota gratuita conta cada texto: 100/min. Lotes de 90 com pausa entre eles.
    for i in range(0, len(texts), EMBED_BATCH):
        if i:
            print(f"  {i}/{len(texts)} embeddings; aguardando a cota (65s)...", flush=True)
            time.sleep(65)
        batch = texts[i : i + EMBED_BATCH]
        resp = _retry(
            lambda: client().models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=config.EMBEDDING_DIM
                ),
            ),
            attempts=attempts,
            wait_429=wait_429,
        )
        vecs.extend(e.values for e in resp.embeddings)
    return _normalize(np.array(vecs, dtype=np.float32))


QUOTA_COOLDOWN = 30 * 60  # modelo com cota diária esgotada: não insistir por 30 min
OUTAGE_COOLDOWN = 60  # modelo sobrecarregado/fora do ar: tenta o próximo por 1 min
_cooldown: dict[str, float] = {}  # modelo -> instante (monotonic) em que volta a ser tentado


def _call_once_or_twice(model: str, prompt: str, cfg: types.GenerateContentConfig):
    """Uma nova tentativa só para falhas momentâneas (5xx/rede). 429 não se repete: é cota."""
    for attempt in range(2):
        try:
            return client().models.generate_content(model=model, contents=prompt, config=cfg)
        except errors.APIError as e:
            if e.code not in (500, 503, 504) or attempt:
                raise
        except httpx.TransportError:
            if attempt:
                raise
        time.sleep(1 + random.random())


def generate(prompt: str, system: str, schema=None, temperature: float = 0.2) -> tuple[str, str]:
    """Retorna (texto, modelo_usado). Percorre GEMINI_MODEL e os modelos reserva: a cota gratuita é
    por modelo (poucas chamadas/dia), então cada reserva amplia a capacidade e a disponibilidade."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    models = list(dict.fromkeys([config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]))
    now = time.monotonic()
    ready = [m for m in models if _cooldown.get(m, 0) <= now]
    last: Exception | None = None
    for model in ready or models:  # se todos estiverem em espera, tenta todos mesmo assim
        try:
            resp = _call_once_or_twice(model, prompt, cfg)
            return resp.text or "", model
        except errors.APIError as e:
            _cooldown[model] = now + (QUOTA_COOLDOWN if e.code == 429 else OUTAGE_COOLDOWN)
            last = e
        except httpx.TransportError as e:
            _cooldown[model] = now + OUTAGE_COOLDOWN
            last = e
    raise last
