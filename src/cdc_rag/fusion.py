"""Busca híbrida: funde os rankings do BM25 e da busca densa por Reciprocal Rank Fusion.

RRF usa só a posição de cada artigo em cada ranking (score = soma de 1 / (k + posição)), então
não precisa calibrar escalas diferentes (BM25 vai de 0 a ~15, cosseno de -1 a 1). k = 60 é o
valor do artigo original (Cormack et al., 2009); não foi ajustado no conjunto de avaliação para
não inflar o resultado.
"""

from .retrieval import Hit


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    articles = {}
    for ranking in rankings:
        for pos, hit in enumerate(ranking, start=1):
            key = hit.article.artigo
            scores[key] = scores.get(key, 0.0) + 1 / (k + pos)
            articles[key] = hit.article
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [Hit(articles[a], round(scores[a], 4)) for a in ranked]


class Hybrid:
    def __init__(self, *indexes, k: int = 60, depth: int = 50):
        self.indexes, self.k, self.depth = indexes, k, depth

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return rrf([idx.search(query, k=self.depth) for idx in self.indexes], k=self.k)[:k]
