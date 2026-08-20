import nltk
from pyconcepticon import Concepticon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sentence_transformers import SentenceTransformer


class BOWEncoder(object):
    def __init__(self, concepts, con: Concepticon = None, min_token_count=2, keep_one_hot=False):
        self.min_token_count = min_token_count
        self.keep_one_hot = keep_one_hot
        self.concepts = concepts or []
        self.concept_to_id = {}
        if self.keep_one_hot:
            for concept in concepts:
                self.concept_to_id[concept] = len(self.concept_to_id)
        self.con = con or Concepticon()
        self.con_definitions = {c.gloss: c.definition for c in self.con.conceptsets.values()}
        self.stop_words = None
        self.init_nltk()
        self.token_to_index = None
        self.init_vocabulary()

    def init_nltk(self):
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        self.stop_words = set(stopwords.words('english'))

    def init_vocabulary(self):
        definitions = {c.gloss: self.tokenize_and_prune(c.definition) for c in self.con.conceptsets.values()
                       if c.gloss in self.concepts}
        counter = Counter()
        for bow in definitions.values():
            counter.update(bow)
        vocabulary = {x for x in counter.keys() if counter[x] >= self.min_token_count}
        self.token_to_index = {x: i for i, x in enumerate(vocabulary)}

    def tokenize_and_prune(self, definition):
        return {x for x in word_tokenize(definition.lower())
                                    if x not in self.stop_words and x.isalpha()}

    def encode_one_hot(self, concept):
        vector = len(self.concept_to_id) * [0]
        idx = self.concept_to_id.get(concept)
        if idx is not None:
            vector[idx] = 1
        return vector

    def encode(self, definition):
        tokens = self.tokenize_and_prune(definition)
        vector = len(self.token_to_index) * [0]
        for token in tokens:
            if token in self.token_to_index:
                vector[self.token_to_index[token]] = 1
        return vector

    def encode_concept(self, gloss):
        if gloss not in self.con_definitions:
            return ValueError(f"Concept {gloss} not in Concepticon")

        vector = self.encode(self.con_definitions[gloss])
        if self.keep_one_hot:
            vector = self.encode_one_hot(gloss) + vector

        return vector

    def generate_encoding_matrix(self, concept_to_id):
        input_dim = len(self.token_to_index) + len(self.concept_to_id) if self.keep_one_hot else len(self.token_to_index)
        matrix = len(concept_to_id) * [input_dim * [0]]
        for concept, id in concept_to_id.items():
            matrix[id] = self.encode_concept(concept)

        return matrix


# encoder.generate_encoding_matrix(concept_to_id)
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
