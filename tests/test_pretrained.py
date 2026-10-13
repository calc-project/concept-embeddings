import pytest

from conceptembeddings import ConceptEmbeddings


def test_pretrained(mockcepticon, mock_concept, mock_definition, mock_concepts):
    ce = ConceptEmbeddings(concepticon=mockcepticon)
    emb1 = ce.embed_definition(mock_definition)
    emb2 = ce(mock_concept)
    emb3 = ce.generate_embeddings(mock_concepts)[mock_concept]
    assert len(emb1) == 128
    assert list(emb1) == list(emb2) == list(emb3)


def test_pretrained_custom(mockcepticon, mock_concept, mock_definition, mock_model_dir):
    ce = ConceptEmbeddings(concepticon=mockcepticon, weights_dir=mock_model_dir)
    assert ce.weights.shape == (2, 3)
    assert ce.encoder.encode_concept(mock_concept)
    assert ce.encoder.encode(mock_definition)


def test_encoding_matrix(mockcepticon, mock_concept, mock_concepts):
    ce = ConceptEmbeddings(concepticon=mockcepticon)
    matrix = ce.encoder.generate_encoding_matrix(mock_concepts)
    assert matrix[0] == ce.encoder.encode_concept(mock_concept)

