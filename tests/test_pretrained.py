import pytest

from conceptembeddings.pretrained import ConceptEmbeddings


def test_pretrained(mockcepticon, mock_concept, mock_definition):
    ce = ConceptEmbeddings(concepticon=mockcepticon)
    emb1 = ce.embed_definition(mock_definition)
    emb2 = ce(mock_concept)
    assert len(emb1) == 128
    assert list(emb1) == list(emb2)
