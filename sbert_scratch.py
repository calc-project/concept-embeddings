from sentence_transformers import SentenceTransformer
from pyconcepticon import Concepticon
from tabulate import tabulate


CONCEPTS = ["EYE", "EAR", "NOSE", "DOG", "SUN", "TOOTH", "STAR", "TONGUE", "BONE", "BLOOD", "WATER", "FIRE"]

model = SentenceTransformer("all-mpnet-base-v2")
con = Concepticon()
con_definitions = {c.gloss: c.definition for c in con.conceptsets.values()}


def sbert_similarities(concepts):
    sentences = [con_definitions[c] for c in concepts]
    embeddings = model.encode(sentences)
    return model.similarity(embeddings, embeddings)


sim = sbert_similarities(CONCEPTS).tolist()
print(tabulate(sim, headers=CONCEPTS, showindex=CONCEPTS))
