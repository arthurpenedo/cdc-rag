from types import SimpleNamespace

import pytest

from cdc_rag.answer import answer, build_documents
from cdc_rag.evaluation import evaluate, load_questions
from cdc_rag.retrieval import BM25, load_articles, stem, tokenize

BASELINE_HIT_AT_3 = 0.55  # linha de base do BM25 nas 58 perguntas (56,9%); a busca não pode piorar


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


def test_pergunta_sem_artigos_nao_chama_o_modelo():
    from cdc_rag.answer import answer_ollama

    vazio = SimpleNamespace(search=lambda _q, k=4: [])

    def nunca(*_a, **_k):
        raise AssertionError("o modelo não deveria ser chamado sem artigos recuperados")

    r1 = answer("xyz", vazio, client=SimpleNamespace(beta=nunca))
    r2 = answer_ollama("xyz", vazio, post=nunca)
    for r in (r1, r2):
        assert "Não encontrei artigos" in r.texto and r.artigos_citados == [] and r.artigos_recuperados == []


def test_recusa_do_modelo_vira_erro_claro(index):
    from cdc_rag.answer import RefusalError

    response = SimpleNamespace(stop_reason="refusal", content=[])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **_kw: response)))
    with pytest.raises(RefusalError, match="recusou"):
        answer("Fui cobrado indevidamente, recebo em dobro?", index, client=client)


def test_ollama_answer_keeps_only_citations_that_were_provided(index):
    import json

    from cdc_rag.answer import answer_ollama

    saida = {"resposta": "Sim, em dobro.", "artigos_citados": ["Art. 42", "Art. 999", "Art. 42"]}
    calls = []

    def post(url, payload):
        calls.append((url, payload))
        return {"message": {"content": json.dumps(saida)}}

    result = answer_ollama("Fui cobrado indevidamente, recebo em dobro?", index, post=post)
    assert result.texto == "Sim, em dobro."
    assert result.artigos_citados == ["Art. 42"] and result.artigos_descartados == ["Art. 999"]
    url, payload = calls[0]
    assert url.endswith("/api/chat") and '<artigo id="Art. 42">' in payload["messages"][1]["content"]


def test_exemplos_page_flags_discarded_citations():
    from cdc_rag.answer import Answer
    from cdc_rag.report import render_exemplos

    html = render_exemplos([("P <b>?", Answer("R", ["Art. 1"], ["Art. 1"], ["Art. 9"]))], "qwen2.5:3b")
    assert "P &lt;b&gt;?" in html and "Citações descartadas" in html and "Art. 9" in html


def test_post_json_envia_json_e_le_resposta(monkeypatch):
    import io
    import json

    from cdc_rag import answer as mod

    vistos = {}

    def falso_urlopen(req, timeout):
        vistos["url"], vistos["corpo"], vistos["timeout"] = req.full_url, json.loads(req.data), timeout
        return io.StringIO('{"ok": true}')

    monkeypatch.setattr(mod.urllib.request, "urlopen", falso_urlopen)
    assert mod._post_json("http://x/api", {"a": 1}) == {"ok": True}
    assert vistos["corpo"] == {"a": 1} and vistos["url"] == "http://x/api" and vistos["timeout"] == 600
