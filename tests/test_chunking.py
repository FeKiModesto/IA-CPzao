from app.chunking import chunk_page, page_url

SAMPLE = """---
title: x
---
# Lab 4 - RAG

Introdução ao laboratório de RAG com bastante texto para passar do tamanho mínimo de um trecho, \
explicando o objetivo geral e o que será construído ao longo da aula prática com embeddings.

## Embeddings

Um embedding é uma lista de números que representa o significado de um texto. Textos relacionados \
ficam próximos no espaço vetorial, o que permite busca semântica em vez de busca por palavra exata.

```python
# # isto não é um título
x = 1
```

## Curto

Pouco.
"""


def test_page_url_matches_mkdocs_directory_urls():
    base = "https://arnaldojr.github.io/DisruptiveArchitectures/"
    assert page_url("aulas/genAI/lab3_5/lab3_5.md") == base + "aulas/genAI/lab3_5/lab3_5/"
    assert page_url("aulas/iot/lab1/index.md") == base + "aulas/iot/lab1/"
    assert page_url("index.md") == base


def test_chunks_keep_title_and_section_trail():
    chunks = chunk_page("aulas/genAI/lab4/lab4.md", SAMPLE)
    assert chunks and all(c.title == "Lab 4 - RAG" for c in chunks)
    emb = next(c for c in chunks if "Embeddings" in c.section)
    assert "embedding é uma lista" in emb.text
    assert emb.embed_text().startswith("Lab 4 - RAG > ")


def test_code_fence_comment_is_not_a_heading():
    chunks = chunk_page("a.md", SAMPLE)
    assert not any(c.section.endswith("isto não é um título") for c in chunks)
    assert any("x = 1" in c.text for c in chunks)


def test_short_section_is_merged_not_dropped():
    chunks = chunk_page("a.md", SAMPLE)
    assert any("Pouco." in c.text for c in chunks)


def test_long_section_is_split_under_limit():
    body = "\n\n".join(f"Parágrafo {i} " + "palavra " * 60 for i in range(20))
    chunks = chunk_page("long.md", f"# T\n\n## S\n\n{body}")
    assert len(chunks) > 1
    assert all(len(c.text) <= 2100 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
