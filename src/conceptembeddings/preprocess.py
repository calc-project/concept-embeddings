from typing import Optional

from pyconcepticon import Concepticon
from sentence_transformers import SentenceTransformer


class SBertEncoder:
    """
    An encoder that generates feature representations for concepts based on their definitions.
    It wraps a Concepticon instance and a SentenceTransformer model. Feature representations are generated
    by retrieving concept definitions from Concepticon and running them through the SentenceTransformer model.
    """
    def __init__(self, concepts, lm_name="all-mpnet-base-v2", con: Optional[Concepticon] = None):
        self.concepts = concepts
        self.con = con or Concepticon()
        self.con_definitions = {c.gloss: c.definition for c in self.con.conceptsets.values()}
        self.model_name = lm_name
        self.model = SentenceTransformer(self.model_name).requires_grad_(False)
        self.encodings = {}

    def encode(self, definition):
        """
        Encodes a concept definition.
        :param definition: a definition of the concept to be encoded.
        :return: the feature representation for the given concept definition.
        """
        return self.model.encode(definition).tolist()

    def encode_concept(self, gloss):
        """
        Encodes a concept defined in Concepticon.
        :param gloss: The concept to be encoded.
        :return: The feature representation for the given concept.
        """
        emb = self.encodings.get(gloss) or self.model.encode(self.con_definitions[gloss]).tolist()
        self.encodings[gloss] = emb
        return emb

    def generate_encoding_matrix(self, concept_to_id):
        """
        Generates an encoding matrix for the given concepts.
        :param concept_to_id: a dict mapping concept name to concept ID.
        :return: An encoding matrix such that each row corresponds to the encoding of the concept at that index.
        """
        matrix = len(concept_to_id) * [None]
        for concept, id in concept_to_id.items():
            matrix[id] = self.encode_concept(concept)

        return matrix
