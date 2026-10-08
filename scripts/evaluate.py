"""Mede a qualidade da recuperação e da resposta com o conjunto eval/questions.json.

  python scripts/evaluate.py            só recuperação (rápido): hit@k e MRR
  python scripts/evaluate.py --full     roda o RAG completo e imprime respostas p/ revisão

Perguntas com "expect": null estão fora do escopo; o assistente deve responder
"não encontrei" (found=false).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import llm, rag  # noqa: E402
from app.config import ROOT  # noqa: E402
from app.retrieval import Index  # noqa: E402


def matches(hit_chunk: dict, expect: list[str]) -> bool:
    haystack = hit_chunk["source"].lower()
    return any(e.lower() in haystack for e in expect)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()

    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    index = Index.load()

    in_scope = [q for q in questions if q["expect"]]
    out_scope = [q for q in questions if not q["expect"]]

    hits_at_k, rr = 0, 0.0
    print(f"== Recuperação (k={args.k}) ==")
    for item in in_scope:
        qvec = llm.embed([item["q"]], task="RETRIEVAL_QUERY")[0]
        hits = index.search(item["q"], qvec, k=args.k)
        rank = next((i for i, h in enumerate(hits, 1) if matches(h.chunk, item["expect"])), None)
        hits_at_k += rank is not None
        rr += 1 / rank if rank else 0
        mark = f"#{rank}" if rank else "MISS"
        top = hits[0].chunk["source"] if hits else "-"
        print(f"  {mark:5} cos={hits[0].score:.2f} {item['q'][:55]:55} -> {top}")
    n = len(in_scope)
    print(f"hit@{args.k} = {hits_at_k}/{n} ({hits_at_k / n:.0%})   MRR = {rr / n:.2f}\n")

    if not args.full:
        print("== Fora do escopo: similaridade máxima (para calibrar MIN_SIMILARITY) ==")
        for item in out_scope:
            qvec = llm.embed([item["q"]], task="RETRIEVAL_QUERY")[0]
            top = max(h.score for h in index.search(item["q"], qvec, k=args.k))
            print(f"  cos={top:.2f} {item['q']}")
        return 0

    print("== RAG completo ==")
    correct_refusals = 0
    for item in questions:
        res = rag.answer(item["q"], [], index)
        print(f"\nP: {item['q']}\nfound={res.found} top={res.top_score:.2f} fontes={[s['n'] for s in res.sources]}")
        print(res.answer[:600])
        if not item["expect"] and not res.found:
            correct_refusals += 1
    print(f"\nRecusas corretas fora do escopo: {correct_refusals}/{len(out_scope)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
