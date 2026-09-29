# cdc-rag

[![CI](https://github.com/arthurpenedo/cdc-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurpenedo/cdc-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Perguntas e respostas sobre o **Código de Defesa do Consumidor** (Lei 8.078/1990) com **RAG**: busca os artigos relevantes, responde com o Claude e **cita os artigos** que embasam cada afirmação. A qualidade da busca é **medida**, não presumida.

## O problema

"Posso desistir de uma compra online?", "Fui cobrado indevidamente, recebo em dobro?". São dúvidas que milhões de consumidores têm, e a resposta está na lei. Um chatbot genérico pode responder com segurança algo que a lei não diz. Aqui a resposta só pode vir dos artigos recuperados, e cada afirmação aponta o artigo de origem.

## Demo

```text
$ cdc-rag buscar "Posso desistir de uma compra feita pela internet?" -k 3
[  4.96] Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias a contar de sua assinatura...
[  3.63] Art. 54-G. Sem prejuízo do disposto no art. 39 deste Código...
[  3.33] Art. 53. Nos contratos de compra e venda de móveis ou imóveis mediante pagamento em prestações...

$ cdc-rag avaliar
Perguntas: 18
hit@3: 77.8%
MRR: 0.657
```

O comando `cdc-rag perguntar "..."` (com `ANTHROPIC_API_KEY`) devolve a resposta do Claude, a lista de **artigos citados** pela API e os artigos recuperados pela busca. *(Exemplo real de saída entra aqui no M2.)*

## Arquitetura

```
Planalto (HTML) ──► scripts/extrair_cdc.py ──► 119 artigos vigentes (JSONL, sem trechos revogados)
                                                      │
pergunta ──► BM25 (retrieval.py) ──► top-k artigos ───┤
                                                      ▼
                         Claude + blocos `document` com citations: {enabled: true}
                                                      │
                                   resposta + artigos citados (vindos da própria API)
```

- **Corpus:** o extrator baixa o texto compilado do Planalto, **descarta trechos revogados** (que o site mostra riscados) e quebra por artigo, incluindo os artigos com letra (ex.: 54-A a 54-G, superendividamento).
- **Busca:** BM25 implementado do zero, com normalização de acentos, stopwords e radicalização leve em português.
- **Resposta:** usa as **citações nativas da API do Claude**. Cada artigo é um documento, e a API devolve quais trechos de quais documentos sustentam o texto. Nada de pedir ao modelo que "escreva a fonte entre colchetes".
- **Avaliação:** 18 perguntas com os artigos esperados; mede hit@k e MRR. Um teste impede que mudanças na busca piorem a linha de base.

### Decisões técnicas

- **Por que BM25 antes de embeddings?** O corpus tem cerca de 120 documentos com vocabulário jurídico estável. BM25 é explicável, instantâneo e não custa nada. A linha de base medida (hit@3 = 77,8%) é o número que embeddings e busca híbrida terão que superar no M2.
- **Chunk = artigo.** O artigo é a unidade natural de citação jurídica, e é assim que um advogado ou o Procon se refere à lei.
- **Erros documentados.** A avaliação lista as perguntas em que a busca falha (ex.: "Quem é considerado consumidor?" perde para artigos que repetem muito a palavra "consumidor"), o que orienta a próxima iteração.

## Como rodar

```bash
git clone https://github.com/arthurpenedo/cdc-rag && cd cdc-rag
pip install -e ".[dev]"

cdc-rag buscar "cobrança indevida"
cdc-rag avaliar -k 3
cdc-rag perguntar "Qual o prazo para reclamar de um defeito?"   # precisa de ANTHROPIC_API_KEY
python scripts/extrair_cdc.py                                   # reconstrói o corpus a partir do Planalto
pytest -q
```

## Próximos passos

- [ ] Busca híbrida (BM25 + embeddings) comparada pela mesma avaliação
- [ ] Ampliar o conjunto de avaliação para 50+ perguntas reais (ex.: temas mais reclamados no consumidor.gov.br)
- [ ] Avaliar a fidelidade das respostas (toda afirmação tem citação?)
- [ ] Interface de chat web

> ⚖️ Projeto educacional. Não substitui orientação jurídica, Procon ou Defensoria Pública.

---

Feito por [Arthur Penedo](https://github.com/arthurpenedo) · [LinkedIn](https://www.linkedin.com/in/arthuralves-penedo)
