from pathlib import Path
from importlib import resources

import numpy as np
from conceptembeddings.preprocess import SBertEncoder
from pyconcepticon import Concepticon


class ConceptEmbeddings:
    """
    Wrapper class for embedding concepts using pretrained weights.
    """
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
        """
        Retrieve the embedding for the given concept. The concept must be defined in the used version of Concepticon.
        :param concept: A concept gloss defined in Concepticon.
        :return: the embedding for the given concept.
        """
        return np.matmul(self.encoder.encode_concept(concept), self.weights)

    def generate_embeddings(self, concepts):
        """
        Generate embeddings for the given concepts. All concepts must be defined in the used version of Concepticon.
        :param concepts: An iterable of concept glosses defined in Concepticon.
        :return: a dict with the concepts as keys and the embedding as numpy arrays.
        """
        return {c: self(c) for c in concepts}

    def embed_definition(self, definition):
        """
        Generates a concept embedding based on the provided definition.
        :param definition: A short definition of the concept.
        :return: the embedding for the given definition.
        """
        return np.matmul(self.encoder.encode(definition), self.weights)
