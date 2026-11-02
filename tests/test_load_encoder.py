import sys

import pytest

from cdc_rag.dense import load_encoder


def test_sem_model2vec_a_mensagem_diz_qual_extra_instalar(monkeypatch):
    monkeypatch.setitem(sys.modules, "model2vec", None)  # força ImportError no import
    with pytest.raises(RuntimeError, match=r"\[embeddings\]"):
        load_encoder()
