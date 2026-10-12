"""Geração de resposta com Claude usando a funcionalidade nativa de CITAÇÕES da API.

Cada artigo recuperado vai como um bloco `document` com `citations: {enabled: true}`.
A API devolve os trechos citados apontando para o documento de origem, então a
resposta final lista exatamente quais artigos embasaram cada afirmação.
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ConfigDict

from .retrieval import Hit

MODEL = os.getenv("CDC_RAG_MODEL", "claude-opus-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("CDC_RAG_OLLAMA_MODEL", "qwen2.5:3b")

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
    artigos_descartados: list[str] = field(default_factory=list)  # ids citados pelo modelo que não foram fornecidos


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


class _Resposta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resposta: str
    artigos_citados: list[str]


SYSTEM_PROMPT_ABERTO = SYSTEM_PROMPT + """

Os artigos vêm numerados em <artigo id="...">. Responda em JSON com:
- "resposta": o texto para o consumidor;
- "artigos_citados": os ids (exatamente como aparecem em id="...") dos artigos que sustentam a resposta.
Cite só artigos que você realmente usou. Se nenhum artigo responde, deixe a lista vazia."""


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def answer_ollama(question: str, index, k: int = 4, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL,
                  post: Callable[[str, dict], dict] = _post_json) -> Answer:
    """Resposta com modelo aberto (Ollama): grátis e sem chave. Citações validadas no código.

    Diferente do Claude, o Ollama não tem citações nativas: o modelo devolve os ids dos artigos que
    usou e o código só aceita ids que estejam entre os artigos recuperados. Id inventado é descartado
    (e contado em `artigos_descartados`), então a resposta nunca aponta para um artigo que não foi fornecido.
    """
    hits = index.search(question, k=k)
    recuperados = [h.article.titulo for h in hits]
    if not hits:
        return Answer("Não encontrei artigos do CDC relacionados a essa pergunta.", [], [])

    contexto = "\n\n".join(f'<artigo id="{h.article.titulo}">\n{h.article.texto}\n</artigo>' for h in hits)
    response = post(f"{url.rstrip('/')}/api/chat", {
        "model": model, "stream": False,
        "format": _Resposta.model_json_schema(),
        "options": {"temperature": 0, "seed": 7},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT_ABERTO},
                     {"role": "user", "content": f"{contexto}\n\nPergunta: {question}"}],
    })
    saida = _Resposta.model_validate(json.loads(response["message"]["content"]))
    validos = set(recuperados)
    citados = list(dict.fromkeys(a for a in saida.artigos_citados if a in validos))
    descartados = [a for a in saida.artigos_citados if a not in validos]
    return Answer(saida.resposta.strip(), citados, recuperados, descartados)


def answer(question: str, index, k: int = 4, client=None) -> Answer:
    hits = index.search(question, k=k)
    recuperados = [h.article.titulo for h in hits]
    if not hits:
        return Answer("Não encontrei artigos do CDC relacionados a essa pergunta.", [], [])

    if client is None:
        import anthropic  # import preguiçoso: a resposta com Ollama não precisa do SDK

        client = anthropic.Anthropic()
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
