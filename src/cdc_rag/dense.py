"""Busca densa com embeddings locais (model2vec), sem API e sem GPU.

Por que model2vec: são embeddings estáticos destilados de um transformer. Rodam na CPU em
milissegundos, sem torch, e o modelo multilíngue entende "nome sujo" perto de "cadastro de
inadimplentes" — exatamente onde o BM25 erra.

Chunk = inciso/parágrafo, pontuação do artigo = melhor chunk. Artigos longos (ex.: art. 39,
com mais de dez incisos) diluem a média dos vetores se forem codificados inteiros; a avaliação mostrou
+9 pontos de hit@3 com chunks. Cada chunk leva o caput do artigo como contexto.

Instale com: pip install -e ".[embeddings]"
"""

import os
import re
from collections.abc import Callable, Sequence

from .retrieval import Article, Hit

# Nome no Hugging Face ou pasta local (o CI baixa uma revisão fixa, sem os arquivos ONNX).
MODEL = os.getenv("CDC_RAG_EMBEDDINGS", "minishlab/potion-multilingual-128M")

Encoder = Callable[[Sequence[str]], "np.ndarray"]  # noqa: F821

_SPLIT = re.compile(r"\n(?=(?:[IVXLC]+ -|§|Parágrafo))")


def split_chunks(article: Article, head_chars: int = 300) -> list[str]:
    """Divide o artigo em caput + incisos/parágrafos; cada pedaço leva o caput como contexto."""
    parts = [p.strip() for p in _SPLIT.split(article.texto) if p.strip()]
    head = parts[0][:head_chars]
    return [parts[0]] + [f"{head} {p}" for p in parts[1:]]


def load_encoder(model_name: str = MODEL) -> Encoder:
    try:
        from model2vec import StaticModel
    except ImportError as exc:
        raise RuntimeError('Busca densa requer o extra de embeddings: pip install -e ".[embeddings]"') from exc
    return StaticModel.from_pretrained(model_name).encode


class DenseIndex:
    def __init__(self, articles: list[Article], encoder: Encoder | None = None):
        import numpy as np

        self.np = np
        self.articles = articles
        self.encode = encoder or load_encoder()
        chunks, owner = [], []
        for i, article in enumerate(articles):
            for chunk in split_chunks(article):
                chunks.append(chunk)
                owner.append(i)
        self.owner = np.array(owner)
        self.vectors = self._unit(self.encode(chunks))

    def _unit(self, x):
        x = self.np.asarray(x, dtype=self.np.float32)
        return x / self.np.clip(self.np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        sims = self.vectors @ self._unit(self.encode([query]))[0]
        best = self.np.full(len(self.articles), -1.0)
        self.np.maximum.at(best, self.owner, sims)
        ranked = self.np.argsort(-best)[:k]
        return [Hit(self.articles[i], round(float(best[i]), 3)) for i in ranked]
