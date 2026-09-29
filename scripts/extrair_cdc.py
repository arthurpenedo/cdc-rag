"""Baixa o CDC (Lei 8.078/1990) do Planalto e gera data/cdc_artigos.jsonl, um artigo por linha.

Texto de lei é de domínio público no Brasil (Lei 9.610/1998, art. 8º, IV).
Trechos revogados (riscados com <strike>/<s> no site do Planalto) são descartados.

Rodar: python scripts/extrair_cdc.py [caminho_do_html_local]
"""

import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"
OUT = Path(__file__).parent.parent / "src" / "cdc_rag" / "data" / "cdc_artigos.jsonl"

ART = re.compile(r"^Art\.\s*(\d+)(?:-([A-Z]))?\s*[º°o]?\.?", re.MULTILINE)


class TextExtractor(HTMLParser):
    """Extrai o texto visível, ignorando o conteúdo riscado (revogado)."""

    SKIP = {"strike", "s", "del", "script", "style"}
    BLOCK = {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts).replace("\xa0", " ")
        lines = [re.sub(r"[ \t\r]+", " ", line).strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def split_articles(text: str) -> list[dict]:
    # O texto normativo começa no Art. 1º; antes disso é cabeçalho da página.
    matches = list(ART.finditer(text))
    articles = []
    seen = set()
    for i, m in enumerate(matches):
        numero = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        # remove títulos de capítulo/seção que ficam colados ao fim do artigo anterior
        body = re.split(r"\n(?:TÍTULO|CAPÍTULO|SEÇÃO|Seção)\s", body)[0].strip()
        if numero in seen or len(body) < 20:
            continue
        seen.add(numero)
        articles.append({"artigo": numero, "texto": body})
    return articles


def main() -> None:
    if len(sys.argv) > 1:
        html = Path(sys.argv[1]).read_bytes()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=60).read()
    parser = TextExtractor()
    parser.feed(html.decode("cp1252", errors="replace"))
    articles = split_articles(parser.text())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in articles) + "\n", encoding="utf-8")
    print(f"{len(articles)} artigos escritos em {OUT}")


if __name__ == "__main__":
    main()
