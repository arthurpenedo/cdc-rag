"""Avaliação da busca: para cada pergunta, o artigo esperado aparece entre os k primeiros?

Métricas:
- hit@k: fração de perguntas com pelo menos um artigo esperado no top-k;
- MRR: média de 1/posição do primeiro artigo esperado encontrado (top-10).

As perguntas estão em linguagem de consumidor ("nome sujo", "boleto", "letras miúdas"), não
copiadas da lei, para medir a busca no uso real e não a coincidência de palavras.
"""

import json
from pathlib import Path

from .retrieval import BM25, Article

QUESTIONS = Path(__file__).parent / "data" / "perguntas_avaliacao.jsonl"
METHODS = ("bm25", "denso", "hibrido")


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(method: str, articles: list[Article], encoder=None):
    """bm25 não tem dependências; denso e hibrido exigem o extra [embeddings] (ou um encoder injetado)."""
    bm25 = BM25(articles)
    if method == "bm25":
        return bm25
    from .dense import DenseIndex

    dense = DenseIndex(articles, encoder=encoder)
    if method == "denso":
        return dense
    if method == "hibrido":
        from .fusion import Hybrid

        return Hybrid(bm25, dense)
    raise ValueError(f"método desconhecido: {method}")


def evaluate(index, questions: list[dict], k: int = 3) -> dict:
    hits, rr, misses, detalhes = 0, 0.0, [], []
    for q in questions:
        ranked = [h.article.artigo for h in index.search(q["pergunta"], k=10)]
        positions = [ranked.index(a) + 1 for a in q["artigos"] if a in ranked]
        first = min(positions) if positions else None
        if first and first <= k:
            hits += 1
        else:
            misses.append({"pergunta": q["pergunta"], "esperado": q["artigos"], "top": ranked[:k]})
        rr += 1 / first if first else 0.0
        detalhes.append({"pergunta": q["pergunta"], "esperado": q["artigos"], "posicao": first, "top": ranked[:k]})
    n = len(questions)
    return {"perguntas": n, "k": k, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3),
            "erros": misses, "detalhes": detalhes}
