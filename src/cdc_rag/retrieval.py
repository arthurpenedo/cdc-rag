"""Busca lexical BM25 sobre os artigos do CDC, implementada do zero (sem dependências).

BM25 é a linha de base: explicável, instantâneo e sem custo. A busca densa (dense.py) e a
híbrida (fusion.py) são comparadas contra ela pela mesma avaliação (evaluation.py).
"""

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "cdc_artigos.jsonl"

STOPWORDS = set("""
a o as os um uma uns umas de da do das dos e em no na nos nas por pela pelo pelas pelos para
com sem que se ao aos ou como mais menos sua seu suas seus eu me meu minha voce tenho tem ter
pode posso quando qual quais quanto quem isso esta este essa esse foi ser sao sobre ate ja nao
art paragrafo inciso lei
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


SUFFIXES = (("coes", "cao"), ("soes", "sao"), ("oes", "ao"), ("aes", "ao"), ("ais", "al"),
            ("eis", "el"), ("res", "r"), ("mente", ""), ("s", ""))


def stem(token: str) -> str:
    """Radicalização leve para português: plurais e alguns sufixos comuns."""
    for suffix, repl in SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + repl
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", normalize(text)) if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Article:
    artigo: str
    texto: str

    @property
    def titulo(self) -> str:
        return f"Art. {self.artigo}"


@dataclass(frozen=True)
class Hit:
    article: Article
    score: float


def load_articles(path: Path = DATA) -> list[Article]:
    return [Article(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BM25:
    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(a.texto)) for a in articles]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def to_dict(self) -> dict:
        """Índice serializado para a busca no navegador (report.py + static/busca.js)."""
        return {
            "k1": self.k1, "b": self.b, "avg_len": self.avg_len, "idf": self.idf,
            "stopwords": sorted(STOPWORDS), "suffixes": SUFFIXES,
            "docs": [{"artigo": a.artigo, "tf": dict(d), "len": n}
                     for a, d, n in zip(self.articles, self.docs, self.lengths)],
        }

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [Hit(self.articles[i], round(scores[i], 3)) for i in ranked[:k] if scores[i] > 0]
