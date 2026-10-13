import numpy as np
from conceptembeddings.preprocess import SBertEncoder
from pyconcepticon import Concepticon
from pathlib import Path
from importlib import resources


class ConceptEmbeddings(object):
    def __init__(self, weights_dir=None, concepticon=None):
        self.concepticon = concepticon or Concepticon()
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.concepts = [x.gloss for x in self.concepticon.conceptsets.values()]
        self.weights, self.encoder = self._load_weights()

    def _load_weights(self):
        if self.weights_dir is not None:
            # if custom weights are given, open as regular files
            weights_fn = self.weights_dir / "weights.txt"
            encoder_fn = self.weights_dir / "encoder.txt"
            try:
                with open(weights_fn, "r") as f:
                    weights = np.loadtxt(f).transpose()
                with open(encoder_fn, "r") as f:
                    lm_name = f.read().strip()
                encoder = SBertEncoder(self.concepts, con=self.concepticon, lm_name=lm_name)
                return weights, encoder
            except Exception as e:
                print(e)
                print("Proceeding with default weights and encoder...")

        # if no custom weights are given or custom weights can not be loaded, return default values
        with resources.files("conceptembeddings.modelweights").joinpath("weights.txt").open("r") as f:
            weights = np.loadtxt(f).transpose()
        with resources.files("conceptembeddings.modelweights").joinpath("encoder.txt").open("r") as f:
            lm_name = f.read().strip()
        encoder = SBertEncoder(self.concepts, con=self.concepticon, lm_name=lm_name)
        return weights, encoder

    def __call__(self, concept):
        return np.matmul(self.encoder.encode_concept(concept), self.weights)

    def generate_embeddings(self, concepts):
        return {c: self(c) for c in concepts}

    def embed_definition(self, definition):
        return np.matmul(self.encoder.encode(definition), self.weights)
