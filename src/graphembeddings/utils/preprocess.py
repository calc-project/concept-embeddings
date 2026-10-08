from pyconcepticon import Concepticon
from sentence_transformers import SentenceTransformer


class SBertEncoder(object):
    def __init__(self, concepts, lm_name="all-mpnet-base-v2", con: Concepticon = None):
        self.concepts = concepts
        self.con = con or Concepticon()
        self.con_definitions = {c.gloss: c.definition for c in self.con.conceptsets.values()}
        self.model = SentenceTransformer(lm_name).requires_grad_(False)
        self.encodings = {}

    def encode(self, definition):
        return self.model.encode(definition).tolist()

    def encode_concept(self, gloss):
        emb = self.encodings.get(gloss) or self.model.encode(self.con_definitions[gloss]).tolist()
        self.encodings[gloss] = emb
        return emb

    def generate_encoding_matrix(self, concept_to_id):
        matrix = len(concept_to_id) * [None]
        for concept, id in concept_to_id.items():
            matrix[id] = self.encode_concept(concept)

        return matrix
