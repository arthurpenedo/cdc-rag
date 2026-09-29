from types import SimpleNamespace

import pytest

from cdc_rag.answer import answer, build_documents
from cdc_rag.evaluation import evaluate, load_questions
from cdc_rag.retrieval import BM25, load_articles, stem, tokenize

BASELINE_HIT_AT_3 = 0.75  # linha de base do BM25; mudanças na busca não podem piorar isso


@pytest.fixture(scope="module")
def index():
    return BM25(load_articles())


def test_corpus_has_key_articles():
    arts = {a.artigo: a for a in load_articles()}
    assert len(arts) > 100
    assert "7 dias" in arts["49"].texto  # direito de arrependimento
    assert "54-A" in arts  # superendividamento (Lei 14.181/2021)


def test_tokenize_normalizes_and_stems():
    assert tokenize("As Cobranças indevidas!") == ["cobranca", "indevida"]
    assert stem("contratacoes") == "contratacao"


def test_search_finds_right_to_regret(index):
    top = [h.article.artigo for h in index.search("desistir compra internet 7 dias", k=3)]
    assert top[0] == "49"


def test_retrieval_does_not_regress(index):
    report = evaluate(index, load_questions(), k=3)
    assert report["hit@3"] >= BASELINE_HIT_AT_3, report["erros"]


def test_documents_enable_citations(index):
    docs = build_documents(index.search("cobrança indevida em dobro", k=2))
    assert all(d["citations"] == {"enabled": True} and d["title"].startswith("Art. ") for d in docs)


def test_answer_collects_cited_articles(index):
    citation = SimpleNamespace(document_title="Art. 42", cited_text="repetição do indébito")
    response = SimpleNamespace(stop_reason="end_turn", content=[
        SimpleNamespace(type="text", text="Sim, em dobro. ", citations=[citation]),
        SimpleNamespace(type="text", text="Procure o Procon.", citations=None),
    ])
    calls = []
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or response)))
    result = answer("Fui cobrado indevidamente, recebo em dobro?", index, client=client)
    assert result.texto == "Sim, em dobro. Procure o Procon."
    assert result.artigos_citados == ["Art. 42"]
    assert "Art. 42" in result.artigos_recuperados
    assert calls[0]["fallbacks"] == "default"
