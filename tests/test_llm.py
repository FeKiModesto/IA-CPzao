from types import SimpleNamespace

import pytest
from google.genai import errors

from app import config, llm


class FakeModels:
    """falhas: modelo -> código HTTP que ele devolve (429 = cota, 503 = sobrecarga)."""

    def __init__(self, falhas):
        self.falhas, self.calls = falhas, []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if model in self.falhas:
            raise _erro(self.falhas[model])
        return SimpleNamespace(text=f"ok:{model}")


def _erro(code):
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"message": "x"}})


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "GEMINI_MODEL", "principal")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", ["reserva1", "reserva2"])
    llm._cooldown.clear()


def use_fake(monkeypatch, falhas):
    fake = FakeModels(falhas)
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=fake))
    return fake


def test_primary_model_is_used_when_healthy(monkeypatch):
    fake = use_fake(monkeypatch, {})
    assert llm.generate("p", "s") == ("ok:principal", "principal")
    assert fake.calls == ["principal"]


def test_quota_exhausted_goes_to_next_model_without_retrying(monkeypatch):
    fake = use_fake(monkeypatch, {"principal": 429})
    assert llm.generate("p", "s") == ("ok:reserva1", "reserva1")
    assert fake.calls == ["principal", "reserva1"]


def test_exhausted_model_is_skipped_on_following_requests(monkeypatch):
    fake = use_fake(monkeypatch, {"principal": 429})
    llm.generate("p", "s")
    fake.calls.clear()
    llm.generate("p", "s")
    assert fake.calls == ["reserva1"]  # não gasta tempo tentando o que já estourou


def test_overload_retries_once_then_falls_back(monkeypatch):
    fake = use_fake(monkeypatch, {"principal": 503})
    assert llm.generate("p", "s")[1] == "reserva1"
    assert fake.calls == ["principal", "principal", "reserva1"]


def test_chain_walks_until_a_model_works(monkeypatch):
    use_fake(monkeypatch, {"principal": 429, "reserva1": 503})
    assert llm.generate("p", "s")[1] == "reserva2"


def test_raises_when_every_model_fails(monkeypatch):
    use_fake(monkeypatch, {"principal": 429, "reserva1": 429, "reserva2": 429})
    with pytest.raises(errors.ClientError):
        llm.generate("p", "s")
