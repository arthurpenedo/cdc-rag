"""CLI.

    cdc-rag buscar "Posso desistir de uma compra online?" [--metodo hibrido]
    cdc-rag perguntar "Posso desistir de uma compra online?"     # precisa de ANTHROPIC_API_KEY
    cdc-rag avaliar [-k 3] [--metodo bm25]
    cdc-rag comparar [--html site/index.html] [--json resultados.json]

bm25 funciona sem dependências; denso e hibrido precisam de: pip install -e ".[embeddings]"
"""

import argparse
import json
import sys
from pathlib import Path

from .evaluation import METHODS, build_index, evaluate, load_questions
from .retrieval import BM25, load_articles


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="cdc-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("buscar", "perguntar"):
        p = sub.add_parser(name)
        p.add_argument("pergunta")
        p.add_argument("-k", type=int, default=4)
        p.add_argument("--metodo", choices=METHODS, default="bm25")
    p_eval = sub.add_parser("avaliar")
    p_eval.add_argument("-k", type=int, default=3)
    p_eval.add_argument("--metodo", choices=METHODS, default="bm25")
    p_cmp = sub.add_parser("comparar", help="avalia todos os métodos disponíveis e gera o relatório")
    p_cmp.add_argument("-k", type=int, default=3)
    p_cmp.add_argument("--html", type=Path)
    p_cmp.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    articles = load_articles()

    if args.cmd == "comparar":
        return _comparar(articles, args)

    try:
        index = build_index(args.metodo, articles)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.cmd == "buscar":
        for hit in index.search(args.pergunta, k=args.k):
            trecho = hit.article.texto.replace("\n", " ")[:160]
            print(f"[{hit.score:>6.2f}] {trecho}...")
    elif args.cmd == "perguntar":
        from .answer import answer

        result = answer(args.pergunta, index, k=args.k)
        print(result.texto)
        print(f"\nArtigos citados: {', '.join(result.artigos_citados) or '-'}")
        print(f"Artigos recuperados: {', '.join(result.artigos_recuperados)}")
    else:
        report = evaluate(index, load_questions(), k=args.k)
        print(f"Método: {args.metodo}")
        print(f"Perguntas: {report['perguntas']}")
        print(f"hit@{args.k}: {report[f'hit@{args.k}']:.1%}")
        print(f"MRR: {report['mrr']:.3f}")
        for miss in report["erros"]:
            print(f"  ✗ {miss['pergunta']} (esperado {miss['esperado']}, top {miss['top']})")
    return 0


def _comparar(articles, args) -> int:
    from .dense import load_encoder

    questions = load_questions()
    try:
        encoder, metodos = load_encoder(), METHODS
    except RuntimeError as exc:
        print(f"Só BM25: {exc}", file=sys.stderr)
        encoder, metodos = None, ("bm25",)
    relatorios = {m: evaluate(build_index(m, articles, encoder), questions, k=args.k) for m in metodos}

    print(f"{'método':<10} {f'hit@{args.k}':>7} {'MRR':>6}")
    for metodo, r in relatorios.items():
        print(f"{metodo:<10} {r[f'hit@{args.k}']:>7.1%} {r['mrr']:>6.3f}")

    for destino in (args.json, args.html):
        if destino:
            destino.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        resumo = {m: {k: v for k, v in r.items() if k != "detalhes"} for m, r in relatorios.items()}
        args.json.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html:
        from .report import render_html

        args.html.write_text(render_html(BM25(articles), relatorios), encoding="utf-8")
        print(f"Relatório: {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
