"""CLI rápida: BM25 e respostas simuladas (sem Ollama, sem Claude, sem embeddings)."""

import json

from cdc_rag import answer as answer_mod
from cdc_rag import cli
from cdc_rag.answer import Answer


def test_buscar_imprime_trechos(capsys):
    assert cli.main(["buscar", "desistir de compra pela internet", "-k", "2"]) == 0
    linhas = capsys.readouterr().out.strip().splitlines()
    assert len(linhas) == 2 and linhas[0].startswith("[")


def test_perguntar_mostra_citadas_e_descartadas(monkeypatch, capsys):
    falsa = Answer("Pode, em 7 dias.", ["Art. 49"], ["Art. 49", "Art. 6"], ["Art. 999"])
    monkeypatch.setattr(answer_mod, "answer_ollama", lambda *a, **k: falsa)
    assert cli.main(["perguntar", "posso desistir?", "--modelo", "ollama"]) == 0
    saida = capsys.readouterr().out
    assert "Artigos citados: Art. 49" in saida
    assert "Artigos recuperados: Art. 49, Art. 6" in saida
    assert "Citações descartadas: Art. 999" in saida


def test_perguntar_sem_citacao_mostra_traco(monkeypatch, capsys):
    monkeypatch.setattr(answer_mod, "answer", lambda *a, **k: Answer("Não sei.", [], ["Art. 1"], []))
    cli.main(["perguntar", "x"])
    saida = capsys.readouterr().out
    assert "Artigos citados: -" in saida and "descartadas" not in saida


def test_metodo_denso_sem_pacote_devolve_2(monkeypatch, capsys):
    def sem_pacote(*a, **k):
        raise RuntimeError("instale o extra embeddings")

    monkeypatch.setattr(cli, "build_index", sem_pacote)
    assert cli.main(["buscar", "x", "--metodo", "denso"]) == 2
    assert "embeddings" in capsys.readouterr().err


def test_comparar_so_bm25_grava_json(monkeypatch, tmp_path, capsys):
    from cdc_rag import dense

    def sem_encoder():
        raise RuntimeError("sem sentence-transformers")

    monkeypatch.setattr(dense, "load_encoder", sem_encoder)
    destino = tmp_path / "sub" / "r.json"
    assert cli.main(["comparar", "--json", str(destino)]) == 0
    dados = json.loads(destino.read_text(encoding="utf-8"))
    assert list(dados) == ["bm25"] and "mrr" in dados["bm25"]
    assert "Só BM25" in capsys.readouterr().err


def test_avaliar_imprime_metricas(capsys):
    assert cli.main(["avaliar", "-k", "3"]) == 0
    saida = capsys.readouterr().out
    assert "Método: bm25" in saida and "hit@3:" in saida and "MRR:" in saida


def test_exemplos_grava_html_com_respostas_do_ollama(monkeypatch, tmp_path, capsys):
    from cdc_rag.report import EXEMPLOS

    falsa = Answer("Pode, em 7 dias.", ["Art. 49"], ["Art. 49"], ["Art. 999"])
    monkeypatch.setattr(answer_mod, "answer_ollama", lambda *a, **k: falsa)
    destino = tmp_path / "site" / "exemplos.html"
    assert cli.main(["exemplos", "--html", str(destino)]) == 0
    pagina = destino.read_text(encoding="utf-8")
    assert "Art. 49" in pagina and EXEMPLOS[0] in pagina
    assert f"Exemplos: {destino}" in capsys.readouterr().out
