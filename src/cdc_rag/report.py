"""Página HTML autocontida: busca no CDC direto no navegador + comparação dos métodos de busca.

A busca da página é o BM25 (static/busca.js) sobre o índice exportado pelo Python, sem servidor.
A comparação mostra hit@k e MRR de cada método e a posição do artigo esperado pergunta a pergunta,
para que os erros fiquem visíveis e não só a média.
"""

import json
from datetime import datetime
from html import escape
from pathlib import Path

from .retrieval import BM25

JS = (Path(__file__).parent / "static" / "busca.js").read_text(encoding="utf-8")

NOMES = {"bm25": "BM25 (lexical)", "denso": "Embeddings (denso)", "hibrido": "Híbrido (RRF)"}

# Exemplos em que a busca do navegador (BM25) acerta o artigo no 1º lugar; as perguntas em que ela
# erra estão na tabela da página, lado a lado com a busca híbrida.
EXEMPLOS = [
    "Posso desistir de uma compra feita pela internet?",
    "Fui cobrado indevidamente, recebo em dobro?",
    "Qual o prazo para reclamar de um produto com defeito?",
    "A oficina precisa me dar orçamento antes do conserto?",
]

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e4e4e0;
--ok:#1f8a4c;--bad:#c2372e;--warn:#b7791f;--bar:#3a6ff7;--bar-bg:#e9edf8}
@media (prefers-color-scheme:dark){:root{--bg:#131316;--card:#1c1c20;--fg:#ececf0;--muted:#9a9aa3;
--line:#2c2c33;--ok:#4cc38a;--bad:#ff6b61;--warn:#e0a84a;--bar:#6f93ff;--bar-bg:#262a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
a{color:var(--bar)}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi b{display:block;font-size:28px;font-variant-numeric:tabular-nums}
.kpi.best{border-color:var(--bar)}
.bar{height:10px;border-radius:5px;background:var(--bar-bg);overflow:hidden;margin:8px 0 4px}
.bar span{display:block;height:100%;background:var(--bar)}
form{display:flex;gap:8px}input{flex:1;min-width:0;font:inherit;padding:10px 12px;border-radius:10px;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{font:inherit;padding:10px 16px;border-radius:10px;border:0;background:var(--bar);color:#fff;cursor:pointer}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.chips button{background:var(--bar-bg);color:var(--fg);padding:4px 10px;font-size:13px;border-radius:999px}
.res{margin-top:10px}.res details{margin:0}.res summary{cursor:pointer;list-style:none}
.res summary::-webkit-details-marker{display:none}
.res .t{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:8px;font-size:14px}
.score{float:right;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}table{width:100%;border-collapse:collapse}
td,th{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.pos{text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
footer{margin-top:40px;font-size:13px}
"""


def _pos(p: int | None, k: int) -> str:
    if p is None:
        return '<td class="pos bad" title="fora do top-10">—</td>'
    cls = "ok" if p <= k else "warn"
    return f'<td class="pos {cls}">{p}º</td>'


def render_html(index: BM25, relatorios: dict[str, dict], data: datetime | None = None) -> str:
    """`relatorios`: método → saída de evaluate(). Métodos ausentes (sem o extra) são omitidos."""
    data = data or datetime.now()
    metodos = list(relatorios)
    k = relatorios[metodos[0]]["k"]
    melhor = max(metodos, key=lambda m: (relatorios[m][f"hit@{k}"], relatorios[m]["mrr"]))

    kpis = []
    for m in metodos:
        r = relatorios[m]
        hit = r[f"hit@{k}"]
        cls = " best" if m == melhor and len(metodos) > 1 else ""
        kpis.append(
            f'<div class="card kpi{cls}"><span class="muted">{escape(NOMES.get(m, m))}</span>'
            f"<b>{hit:.1%}</b><div class=\"bar\"><span style=\"width:{hit:.1%}\"></span></div>"
            f'<span class="muted">hit@{k} · MRR {r["mrr"]:.3f}</span></div>'
        )

    linhas = []
    for i, d in enumerate(relatorios[metodos[0]]["detalhes"]):
        cells = "".join(_pos(relatorios[m]["detalhes"][i]["posicao"], k) for m in metodos)
        esperado = ", ".join(f"Art. {a}" for a in d["esperado"])
        linhas.append(f"<tr><td>{escape(d['pergunta'])}</td><td class=\"muted\">{escape(esperado)}</td>{cells}</tr>")
    heads = "".join(f'<th style="text-align:center">{escape(NOMES.get(m, m).split(" (")[0])}</th>' for m in metodos)

    indice = index.to_dict()
    indice["textos"] = {a.artigo: a.texto for a in index.articles}
    dados = json.dumps(indice, ensure_ascii=False).replace("</", "<\\/")
    chips = "".join(f'<button type="button">{escape(e)}</button>' for e in EXEMPLOS)
    n = relatorios[metodos[0]]["perguntas"]

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cdc-rag · Busca no CDC</title><style>{CSS}</style></head><body><main>
<h1>Busca no Código de Defesa do Consumidor</h1>
<p class="muted">Encontra os artigos da Lei 8.078/1990 que respondem a uma dúvida de consumidor.
Projeto <a href="https://github.com/arthurpenedo/cdc-rag">cdc-rag</a>: RAG com citação dos artigos e busca medida.</p>

<h2>Experimente</h2>
<div class="card">
<form id="f"><input id="q" placeholder="Escreva sua dúvida, ex.: comprei online e me arrependi" aria-label="Pergunta">
<button>Buscar</button></form>
<div class="chips" id="chips">{chips}</div>
<div id="out"></div>
<p class="muted" style="margin:12px 0 0;font-size:13px">Esta caixa roda a busca lexical (BM25) no seu navegador,
sem servidor. A busca híbrida usa um modelo de embeddings e roda no Python; o desempenho de cada uma está abaixo.</p>
</div>

<h2>Comparação dos métodos de busca</h2>
<p class="muted">{n} perguntas em linguagem de consumidor, cada uma com o(s) artigo(s) que a responde(m).
hit@{k} = o artigo certo aparece entre os {k} primeiros; MRR = média de 1/posição.</p>
<div class="kpis">{''.join(kpis)}</div>

<h2>Pergunta a pergunta</h2>
<p class="muted">Posição do primeiro artigo esperado em cada método:
<span class="ok">verde</span> = top-{k}, <span class="warn">amarelo</span> = top-10, <span class="bad">—</span> = não encontrado.</p>
<div class="card wrap"><table><thead><tr><th>Pergunta</th><th>Esperado</th>{heads}</tr></thead>
<tbody>{''.join(linhas)}</tbody></table></div>

<footer class="muted">⚖️ Projeto educacional; não substitui orientação jurídica, Procon ou Defensoria Pública.
Texto da lei: planalto.gov.br. Gerado em {data:%d/%m/%Y %H:%M}. Feito por
<a href="https://github.com/arthurpenedo">Arthur Penedo</a>.</footer>
</main>
<script id="indice" type="application/json">{dados}</script>
<script>{JS}
const indice = JSON.parse(document.getElementById("indice").textContent);
const busca = criarBusca(indice);
const out = document.getElementById("out"), q = document.getElementById("q");
function esc(s) {{ return s.replace(/[&<>"]/g, (c) => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}})[c]); }}
function buscar() {{
  const hits = busca.search(q.value, 5);
  if (!q.value.trim()) {{ out.innerHTML = ""; return; }}
  if (!hits.length) {{ out.innerHTML = '<p class="muted">Nenhum artigo encontrado. Tente outras palavras.</p>'; return; }}
  out.innerHTML = hits.map((h, i) => {{
    const t = indice.textos[h.artigo];
    return `<div class="card res"><details${{i === 0 ? " open" : ""}}><summary><span class="score">${{h.score.toFixed(2)}}</span>`
      + `<strong>Art. ${{esc(h.artigo)}}</strong> <span class="muted">${{esc(t.replace(/^Art\\.\\s*\\S+\\s*/, "").slice(0, 130))}}…</span></summary>`
      + `<div class="t">${{esc(t)}}</div></details></div>`;
  }}).join("");
}}
document.getElementById("f").addEventListener("submit", (e) => {{ e.preventDefault(); buscar(); }});
document.getElementById("chips").addEventListener("click", (e) => {{
  if (e.target.tagName === "BUTTON") {{ q.value = e.target.textContent; buscar(); }}
}});
const inicial = new URLSearchParams(location.search).get("q");
if (inicial) {{ q.value = inicial; buscar(); }}
</script></body></html>
"""
