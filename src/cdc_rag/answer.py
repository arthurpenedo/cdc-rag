"""Geração de resposta com Claude usando a funcionalidade nativa de CITAÇÕES da API.

Cada artigo recuperado vai como um bloco `document` com `citations: {enabled: true}`.
A API devolve os trechos citados apontando para o documento de origem, então a
resposta final lista exatamente quais artigos embasaram cada afirmação.
"""

import os
from dataclasses import dataclass, field

import anthropic

from .retrieval import BM25, Hit

MODEL = os.getenv("CDC_RAG_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """Você responde dúvidas de consumidores brasileiros com base EXCLUSIVAMENTE nos artigos \
do Código de Defesa do Consumidor fornecidos como documentos.

- Responda em português simples, em até 3 parágrafos curtos.
- Se os artigos fornecidos não respondem à pergunta, diga isso claramente em vez de supor.
- Não dê aconselhamento jurídico individual; ao final, se for o caso, sugira procurar o Procon \
ou o consumidor.gov.br."""


class RefusalError(RuntimeError):
    pass


@dataclass
class Answer:
    texto: str
    artigos_citados: list[str] = field(default_factory=list)
    artigos_recuperados: list[str] = field(default_factory=list)


def build_documents(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.article.texto},
            "title": h.article.titulo,
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def answer(question: str, index: BM25, k: int = 4, client: anthropic.Anthropic | None = None) -> Answer:
    hits = index.search(question, k=k)
    recuperados = [h.article.titulo for h in hits]
    if not hits:
        return Answer("Não encontrei artigos do CDC relacionados a essa pergunta.", [], [])

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [*build_documents(hits), {"type": "text", "text": question}]}],
    )
    if response.stop_reason == "refusal":
        raise RefusalError("O modelo recusou responder a esta pergunta.")

    texto, citados = [], []
    for block in response.content:
        if block.type != "text":
            continue
        texto.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            title = getattr(citation, "document_title", None)
            if title and title not in citados:
                citados.append(title)
    return Answer("".join(texto).strip(), citados, recuperados)
