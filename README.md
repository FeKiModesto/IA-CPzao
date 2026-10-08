# DisruptiveBot — Assistente RAG da Disruptive Architectures

Chat com **RAG** sobre o site da matéria [Disruptive Architectures](https://arnaldojr.github.io/DisruptiveArchitectures/).
Responde dúvidas sobre as aulas de IA e IoT usando **somente** o conteúdo do site, cita as páginas usadas e
diz "não encontrei" quando a informação não está no material.

**CPzão (CP5 + CP6) — FIAP ADS 2TDSPG - 2026**

**Demo ao vivo:** https://assistente-disruptive.onrender.com

> ⚠️ O serviço usa o plano gratuito do Render: após inatividade pode levar até 50 segundos para responder a primeira requisição.

## Integrantes

- Felipe Kirschner Modesto (RM561810)
- João Victor Luiz Oliveira Resende (RM565139)
- Pedro Vaz Ferreira (RM566551)
- Vitor Dias dos Santos (RM565422)

## O que é o DisruptiveBot?

Um assistente de IA que responde perguntas sobre o conteúdo da disciplina usando **RAG (Retrieval-Augmented Generation)**: antes de gerar cada resposta, o sistema busca os trechos mais relevantes do site do professor e usa apenas esse contexto para fundamentar a resposta. Se a informação não estiver no material, o bot informa em vez de inventar.

## Arquitetura

```
                 ┌──────────────── offline (uma vez) ────────────────┐
  site do prof.  │ fetch_docs.py → chunking.py → ingest.py (embeddings)│
  (.md no GitHub)│                                   ↓                 │
                 │                         data/index/ (commitado)     │
                 └─────────────────────────────┬───────────────────────┘
                                               │ carregado no startup
 Navegador ──► FastAPI ──► rag.py ──► retrieval.py (busca híbrida) ──► Gemini
 (index.html)  main.py        │                                          (gera resposta
                  │           └── reescreve pergunta de follow-up         em JSON c/ fontes)
                  ▼
        SQLite (local) / Postgres (Render): interações, fontes, latência, erros, feedback
```

| Arquivo | Responsabilidade |
|---|---|
| `app/main.py` | API HTTP (FastAPI), rate limit, orquestra banco + RAG |
| `app/rag.py` | Pipeline: reescrita da pergunta → busca → prompt → resposta estruturada |
| `app/retrieval.py` | Índice em memória: cosseno + BM25 fundidos por RRF |
| `app/chunking.py` | Markdown → trechos por título, preservando blocos de código |
| `app/llm.py` | Cliente Gemini (embeddings e geração), retry e limites de cota |
| `app/db.py` | Modelo `interactions` e acesso ao banco (SQLAlchemy) |
| `app/static/index.html` | Interface de chat (sem dependências externas) |
| `scripts/` | `fetch_docs.py` (baixa o site), `ingest.py` (gera o índice), `evaluate.py` (avalia) |
| `eval/questions.json` | Perguntas de teste com a página esperada |

## Decisões técnicas

- **Gemini (Google AI Studio):** chave gratuita; mesmo provedor usado nos labs da disciplina. Embeddings com
  `gemini-embedding-001` (768 dimensões, tipos de tarefa distintos para documento e pergunta) e geração com `gemini-3.8-flash`.
- **Busca híbrida (embeddings + BM25 + RRF):** embeddings acham o sentido; BM25 acha termos exatos como
  `MQTT`, `ESP32` ou `Lab 3.5`, que a busca semântica sozinha às vezes perde. Os dois rankings são fundidos por
  Reciprocal Rank Fusion.
- **Índice em NumPy, sem banco vetorial:** o corpus tem ~750 trechos; uma multiplicação de matrizes é instantânea
  e evita um serviço a mais (e RAM extra no plano gratuito). O índice é commitado, então o deploy não recalcula nada.
- **Trechos por título:** cada trecho carrega `Página > Seção` no texto embutido e a URL da página, o que permite citar a fonte.
- **Resposta estruturada (JSON):** o modelo devolve `found`, `answer` e `used_sources`. Assim a interface mostra só as
  fontes realmente usadas e a recusa é um campo explícito, não um texto a interpretar.
- **Reescrita de follow-up:** "e como configuro isso?" é transformado em pergunta autocontida antes da busca, usando o histórico salvo no banco.
- **Anti-alucinação:** prompt restrito ao contexto, recusa explícita, limiar mínimo de similaridade (nem chama o LLM se nada se relaciona)
  e o contexto é tratado como dado, não como instrução (mitiga prompt injection vindo das páginas).
- **Banco com `DATABASE_URL`:** SQLite em desenvolvimento, Postgres em produção. O disco do Render gratuito é efêmero, por isso não usamos SQLite lá.
- **Segurança:** chave só em variável de ambiente (`.env` fora do Git), limite de requisições por IP, tamanho máximo de pergunta,
  saída da interface escapada contra XSS.

## Como rodar localmente

Requisitos: Python 3.12+ e uma chave gratuita do [Google AI Studio](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

copy .env.example .env             # Linux/macOS: cp .env.example .env
# edite o .env e cole sua GEMINI_API_KEY

uvicorn app.main:app --reload
```

Abra http://localhost:8000. O índice (`data/index/`) já vem no repositório. Para **refazer** o índice a partir do site:

```bash
python scripts/fetch_docs.py      # baixa os .md do repositório do professor
python scripts/ingest.py          # gera os embeddings (a cota gratuita leva ~9 min)
```

### Com Docker

```bash
docker build -t assistente .
docker run --rm -p 8000:8000 --env-file .env assistente
```

## API

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/chat` | `{"question": "...", "session_id": "opcional"}` → resposta, fontes, `session_id`, latência |
| `GET` | `/history?session_id=...` | Conversa salva no banco |
| `POST` | `/feedback` | `{"interaction_id", "session_id", "value": 1 \| -1}` |
| `GET` | `/stats` | Total de interações, recusas, erros, feedback, latência média |
| `GET` | `/health` | Status e quantidade de trechos carregados |

```bash
curl -X POST http://localhost:8000/chat \ 
     -H "Content-Type: application/json" \
     -d '{"question": "O que o Lab 3.5 pede para entregar?"}'
```

Documentação interativa em `/docs`: https://assistente-disruptive.onrender.com/docs

## Testes e avaliação

```bash
pytest                              # testes unitários e de API (sem rede, LLM simulado)
python scripts/evaluate.py          # qualidade da recuperação: hit@k e MRR
python scripts/evaluate.py --full   # RAG completo; imprime respostas para revisão
```

Os testes cobrem chunking, busca, persistência (inclusive "reinício" do app), feedback, falha do LLM e rate limit.
A avaliação usa perguntas do material com a página esperada, e perguntas fora do escopo que devem ser recusadas.

**Resultado da avaliação:** _(preencher com a saída de `scripts/evaluate.py`)_

## Deploy (Render)

1. Suba o repositório no GitHub (sem `.env`).
2. No Render: **New → Blueprint** e aponte para o repositório. O `render.yaml` cria o web service (Docker) e o Postgres automaticamente.
3. Na tela de configuração do Blueprint, defina `GEMINI_API_KEY` com sua chave do Google AI Studio.
4. Clique em **Deploy Blueprint** e aguarde (~1 min). Confira o status em `/health`.

Observações do plano gratuito: o serviço hiberna após inatividade (a primeira requisição leva ~30–50 s) e o Postgres gratuito expira em 30 dias.

## Limitações

- O conteúdo vem de uma cópia do site; se o professor atualizar as páginas, é preciso rodar `fetch_docs.py` + `ingest.py` e commitar `data/index/`.
- Só indexa as páginas `.md`; o texto dos notebooks `.ipynb` não entra.
- A cota gratuita do Gemini limita o volume de perguntas simultâneas.

## Referências

- [Site da disciplina — Disruptive Architectures](https://arnaldojr.github.io/DisruptiveArchitectures/)
- [Lab 4 — RAG e Bases de Conhecimento](https://arnaldojr.github.io/DisruptiveArchitectures/aulas/genAI/lab4/lab4/)
- [Lab 3.5 — Do protótipo ao produto](https://arnaldojr.github.io/DisruptiveArchitectures/aulas/genAI/lab3_5/lab3_5/)
- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [Documentação Google GenAI (Gemini)](https://ai.google.dev/gemini-api/docs)
