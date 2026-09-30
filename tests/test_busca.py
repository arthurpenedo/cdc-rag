import json
import shutil
import subprocess
import zlib
from pathlib import Path

import pytest

from cdc_rag.evaluation import build_index, evaluate, load_questions
from cdc_rag.fusion import Hybrid, rrf
from cdc_rag.report import render_html
from cdc_rag.retrieval import BM25, Article, Hit, load_articles

JS = Path(__file__).parents[1] / "src" / "cdc_rag" / "static" / "busca.js"

# Linhas de base medidas nas 58 perguntas; mudanças na busca não podem piorá-las.
BASELINE = {"denso": 0.60, "hibrido": 0.62}


@pytest.fixture(scope="module")
def articles():
    return load_articles()


def fake_encoder(texts):
    """Encoder determinístico (bag of words com hashing) para testar a busca densa sem baixar modelo."""
    np = pytest.importorskip("numpy")
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 64] += 1
    return out


def test_rrf_rewards_agreement():
    a, b, c = (Article(x, x) for x in "abc")
    fused = rrf([[Hit(a, 9), Hit(b, 5)], [Hit(b, 0.9), Hit(c, 0.8)]])
    assert [h.article.artigo for h in fused] == ["b", "a", "c"]  # b aparece nos dois rankings


def test_split_chunks_keeps_head_as_context(articles):
    from cdc_rag.dense import split_chunks

    art39 = next(a for a in articles if a.artigo == "39")
    chunks = split_chunks(art39)
    assert len(chunks) > 10  # caput + incisos da venda casada, produto não solicitado etc.
    assert all(c.startswith("Art. 39.") for c in chunks)


def test_dense_and_hybrid_with_fake_encoder(articles):
    pytest.importorskip("numpy")
    dense = build_index("denso", articles, encoder=fake_encoder)
    top = dense.search("desistir do contrato no prazo de 7 dias", k=3)
    assert len(top) == 3 and top[0].score >= top[1].score
    hybrid = Hybrid(BM25(articles), dense)
    assert hybrid.search("desistir do contrato no prazo de 7 dias", k=1)[0].article.artigo == "49"


@pytest.mark.parametrize("metodo", sorted(BASELINE))
def test_real_embeddings_do_not_regress(articles, metodo):
    pytest.importorskip("model2vec")
    report = evaluate(build_index(metodo, articles), load_questions(), k=3)
    assert report["hit@3"] >= BASELINE[metodo], report["erros"]


def test_report_has_search_and_comparison(articles):
    index = BM25(articles)
    relatorios = {"bm25": evaluate(index, load_questions(), k=3)}
    html = render_html(index, relatorios)
    assert "criarBusca" in html and 'id="indice"' in html
    assert "hit@3" in html and html.count("<tr>") == len(load_questions()) + 1
    embutido = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(embutido)["textos"]["49"].startswith("Art. 49")


@pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")
def test_browser_search_matches_python(articles, tmp_path):
    index = BM25(articles)
    queries = [q["pergunta"] for q in load_questions()[:20]] + ["Cobrança INDEVIDA em dobro!", "ÔNUS da prova"]
    (tmp_path / "indice.json").write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "q.json").write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    script = f"""
const fs = require("fs");
const {{ criarBusca }} = require({json.dumps(str(JS))});
const busca = criarBusca(JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "indice.json"))}, "utf8")));
const qs = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "q.json"))}, "utf8"));
process.stdout.write(JSON.stringify(qs.map((q) => busca.search(q, 5).map((h) => h.artigo))));
"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(out.stdout)
    py = [[h.article.artigo for h in index.search(q, k=5)] for q in queries]
    assert js == py


def test_page_examples_hit_first_place_in_browser_search(articles):
    from cdc_rag.report import EXEMPLOS

    index = BM25(articles)
    esperado = ["49", "42", "26", "40"]
    assert [index.search(q, k=1)[0].article.artigo for q in EXEMPLOS] == esperado
