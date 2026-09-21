import csv
import numpy as np
import matplotlib.pyplot as plt
from pycldf import Dataset
from pathlib import Path
from collections import defaultdict
from skbio.stats.distance import DistanceMatrix, mantel

from graphembeddings.utils.preprocess import SBertEncoder


SAVED_WEIGHTS_DIR = Path(__file__).parent.parent
WEIGHTS_FILE = SAVED_WEIGHTS_DIR / "weights.npy"

class ConceptEmbeddings(object):
    def __init__(self, encoder, weights_fn):
        self.encoder = encoder
        self.weights = self._load_weights(weights_fn)

    def _load_weights(self, weights_fn):
        with open(SAVED_WEIGHTS_DIR / weights_fn, "rb") as f:
            weights = np.load(f).transpose()
        return weights

    def __call__(self, concept):
        return np.matmul(self.encoder.encode_concept(concept), self.weights)

    def generate_embeddings(self, concepts):
        return {c: self(c) for c in concepts}


# tonal markers are usually not written in orthography
slv_replacements = [
    ('à', 'a'),
    ('á', 'a'),
    ('è', 'e'),
    ('é', 'e'),
    ('ê', 'e'),
    ('ì', 'i'),
    ('í', 'i'),
    ('ò', 'o'),
    ('ó', 'o'),
    ('ô', 'o'),
    ('ù', 'u'),
    ('ú', 'u'),
    ('ŕ', 'r'),
    ('̀', '')
]

# read in forms -> concept for slovenian from northeuralex
base_dir = Path(__file__).parent
nelex = Dataset.from_metadata(base_dir / "northeuralex" / "cldf" / "cldf-metadata.json")
slv_words = defaultdict(list)
scope_concepts = []
for row in nelex.iter_rows("FormTable"):
    if row["Language_ID"] != "slv":
        continue
    concept = nelex.get_row("ParameterTable", row["Parameter_ID"])["Concepticon_Gloss"]
    form = row["Value"]
    for source, target in slv_replacements:
        form = form.replace(source, target)
    if concept:
        slv_words[form].append(concept)
        scope_concepts.append(concept)


association_counts = defaultdict(lambda: defaultdict(int))

with open(base_dir / "SWOW-SL24" / "SWOW-SL1.0_responses.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        cue = row["cue"]
        # only include cue words that can be linked to Concepticon concepts
        if cue in slv_words and slv_words[cue]:
            # TODO check literature whether to take R1 or rather R123
            for i in range(3):
                response = row[f"response{i+1}Normalized"]
                if response:
                    association_counts[cue][response] += 1

word_to_id = {w: i for i, w in enumerate(association_counts)}
for count_dict in association_counts.values():
    for response in count_dict:
        if response not in word_to_id:
            word_to_id[response] = len(word_to_id)

assoc_counts_matrix = np.zeros((len(association_counts), len(word_to_id)))
for cue, responses in association_counts.items():
    i = word_to_id[cue]
    for response, count in responses.items():
        j = word_to_id[response]
        assoc_counts_matrix[i, j] = count

# based on the words in the target language, create concept embedding matrix
concept_embeddings_by_word = []
encoder = SBertEncoder(scope_concepts)
concept_embedder = ConceptEmbeddings(encoder, WEIGHTS_FILE)

for word in association_counts:
    concepts = slv_words[word]
    embeddings = []
    for concept in concepts:
        embedding = concept_embedder(concept)
        # embedding = encoder.encode_concept(concept)
        embeddings.append(embedding)
    concept_embeddings_by_word.append(np.mean(embeddings, axis=0))

concept_embeddings_by_word = np.array(concept_embeddings_by_word)


def create_distance_matrix(matrix):
    distances = np.zeros((len(matrix), len(matrix)))
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            vec1 = matrix[i]
            vec2 = matrix[j]
            distances[i, j] = distances[j, i] = (1 -
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return DistanceMatrix(distances)


assoc_distances = create_distance_matrix(assoc_counts_matrix)
embeddings_distances = create_distance_matrix(concept_embeddings_by_word)

print(mantel(assoc_distances, embeddings_distances, method="pearson"))
plt.scatter(assoc_distances.condensed_form(), embeddings_distances.condensed_form())
plt.show()
