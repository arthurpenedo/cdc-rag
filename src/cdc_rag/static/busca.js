// BM25 no navegador: mesma normalização, stopwords e radicalização de retrieval.py.
// O índice (idf, frequências, stopwords, sufixos) vem do Python via BM25.to_dict(), então
// só a lógica é duplicada aqui; tests/test_busca_js.py confere que os dois lados dão o mesmo ranking.
function criarBusca(indice) {
  const stopwords = new Set(indice.stopwords);

  function normalize(text) {
    return text.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
  }

  function stem(token) {
    for (const [suffix, repl] of indice.suffixes) {
      if (token.length > suffix.length + 3 && token.endsWith(suffix)) {
        return token.slice(0, -suffix.length) + repl;
      }
    }
    return token;
  }

  function tokenize(text) {
    return (normalize(text).match(/[a-z0-9]+/g) || [])
      .filter((t) => !stopwords.has(t) && t.length > 1)
      .map(stem);
  }

  function search(query, k = 5) {
    const terms = tokenize(query);
    const { k1, b, avg_len: avgLen, idf } = indice;
    const scored = indice.docs.map((doc, i) => {
      let s = 0;
      for (const t of terms) {
        const tf = doc.tf[t] || 0;
        if (tf) s += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc.len / avgLen));
      }
      return { i, s };
    });
    scored.sort((x, y) => y.s - x.s);
    return scored.slice(0, k).filter((x) => x.s > 0)
      .map((x) => ({ artigo: indice.docs[x.i].artigo, score: Math.round(x.s * 1000) / 1000 }));
  }

  return { tokenize, search };
}

if (typeof module !== "undefined") module.exports = { criarBusca };
