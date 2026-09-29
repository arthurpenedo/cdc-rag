"""CLI.

    cdc-rag buscar "Posso desistir de uma compra online?"
    cdc-rag perguntar "Posso desistir de uma compra online?"     # precisa de ANTHROPIC_API_KEY
    cdc-rag avaliar [-k 3]
"""

import argparse
import sys

from .evaluation import evaluate, load_questions
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
    p_eval = sub.add_parser("avaliar")
    p_eval.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)

    index = BM25(load_articles())

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
        print(f"Perguntas: {report['perguntas']}")
        print(f"hit@{args.k}: {report[f'hit@{args.k}']:.1%}")
        print(f"MRR: {report['mrr']:.3f}")
        for miss in report["erros"]:
            print(f"  ✗ {miss['pergunta']} (esperado {miss['esperado']}, top {miss['top']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
