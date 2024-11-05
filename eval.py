import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from newick import loads
from pylocluster import linkage
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr


# load trained embeddings
embeddings = {}

with open("clics-embeddings.tsv") as f:
    for line in f:
        concept, embedding = line.strip().split("\t")
        embedding = np.array(eval(embedding))
        embeddings[concept] = embedding

# load multisimlex ratings
msl = {}

with open("multisimlex.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        concept1 = row["CGL1"]
        concept2 = row["CGL2"]
        mean_similarity = float(row["mean"])
        if concept1 != concept2:
            msl[(concept1, concept2)] = mean_similarity

msl_similarities = []
pred_similarities = []

for concept_pair, similarity in msl.items():
    c1, c2 = concept_pair
    if c1 in embeddings and c2 in embeddings:
        emb1 = embeddings[c1]
        emb2 = embeddings[c2]
        pred_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        msl_similarities.append(similarity)
        pred_similarities.append(pred_similarity)

msl_similarities = np.array(msl_similarities)
pred_similarities = np.array(pred_similarities)
# plt.scatter(msl_similarities, pred_similarities)
# plt.show()

print("Spearman:", spearmanr(msl_similarities, pred_similarities))
print("Pearson:", pearsonr(msl_similarities, pred_similarities))

sys.exit(0)
###############################################

# example words
words = ["ARM", "HAND", "LEG", "FOOT", "HEAD", "SKULL", "BONE", "FIRE", "WOOD", "TREE"]

# set up distance matrix (distance = cosine distance)
distance_matrix = np.zeros((len(words), len(words)))

for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            emb1 = embeddings[word1]
            emb2 = embeddings[word2]
            cos_distance = 1 - (np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            distance_matrix[i, j] = distance_matrix[j, i] = cos_distance

print(tabulate(distance_matrix, headers=words, showindex=words, floatfmt=".4f"))

print(loads(linkage(distance_matrix, taxa=words))[0].ascii_art())

# set up embedding matrix for specified words
matrix = np.array([embeddings[word] for word in words])

# PCA
pca = PCA(n_components=2)
res = pca.fit_transform(matrix)
plt.scatter(*np.swapaxes(res, 0, 1))
for concept, coordinates in zip(words, res):
    plt.annotate(concept, coordinates)
plt.title("PCA")
plt.show()

plt.cla()

# TSNE
tsne = TSNE(n_components=2, perplexity=2)
res = tsne.fit_transform(matrix)
plt.scatter(*np.swapaxes(res, 0, 1))
for concept, coordinates in zip(words, res):
    plt.annotate(concept, coordinates)
plt.title("TSNE")
plt.show()
