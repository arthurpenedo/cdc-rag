"""Avaliação da busca: para cada pergunta, o artigo esperado aparece entre os k primeiros?

Métricas:
- hit@k: fração de perguntas com pelo menos um artigo esperado no top-k;
- MRR: média de 1/posição do primeiro artigo esperado encontrado.
"""

import json
from pathlib import Path

from .retrieval import BM25

QUESTIONS = Path(__file__).parent / "data" / "perguntas_avaliacao.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(index: BM25, questions: list[dict], k: int = 3) -> dict:
    hits, rr, misses = 0, 0.0, []
    for q in questions:
        ranked = [h.article.artigo for h in index.search(q["pergunta"], k=10)]
        positions = [ranked.index(a) + 1 for a in q["artigos"] if a in ranked]
        first = min(positions) if positions else None
        if first and first <= k:
            hits += 1
        else:
            misses.append({"pergunta": q["pergunta"], "esperado": q["artigos"], "top": ranked[:k]})
        rr += 1 / first if first else 0.0
    n = len(questions)
    return {"perguntas": n, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3), "erros": misses}
