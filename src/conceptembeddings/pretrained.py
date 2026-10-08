import numpy as np
from conceptembeddings.preprocess import SBertEncoder
from pyconcepticon import Concepticon
from pathlib import Path


class ConceptEmbeddings(object):
    def __init__(self, weights_fn=None, encoder=None, concepticon=None):
        weights_fp = Path(weights_fn) if weights_fn else Path(__file__).parent / "weights.npy"
        self.concepticon = concepticon or Concepticon()
        concepts = [x.gloss for x in self.concepticon.conceptsets.values()]
        self.encoder = encoder or SBertEncoder(concepts, con=self.concepticon)
        self.weights = self._load_weights(weights_fp)

    def _load_weights(self, weights_fn):
        if weights_fn.suffix == ".txt":
            with open(weights_fn, "r") as f:
                weights = np.loadtxt(f).transpose()
        else:
            with open(weights_fn, "rb") as f:
                weights = np.load(f).transpose()
        return weights

    def __call__(self, concept):
        return np.matmul(self.encoder.encode_concept(concept), self.weights)

    def generate_embeddings(self, concepts):
        return {c: self(c) for c in concepts}

    def embed_definition(self, definition):
        return np.matmul(self.encoder.encode(definition), self.weights)
