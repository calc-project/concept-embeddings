import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from newick import loads
from pylocluster import linkage
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
from pyconcepticon import Concepticon


# load trained embeddings
embeddings = {}

with open("embeddings/2025-01-07/n2v-sg-c.tsv") as f:
    for line in f:
        concept, embedding = line.strip().split("\t")
        embedding = np.array(eval(embedding))
        embeddings[concept] = embedding

# load multisimlex ratings
msl = {}
pl_dict = {}

with open("multisimlex.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        concept1 = row["CGL1"]
        concept2 = row["CGL2"]
        mean_similarity = float(row["mean"])
        path_length = float(row["shortest_path_weighted"]) if row["shortest_path_weighted"] else np.nan
        if concept1 != concept2:
            msl[(concept1, concept2)] = mean_similarity
            pl_dict[(concept1, concept2)] = path_length

msl_similarities = []
pred_similarities = []
path_lengths = []

for concept_pair, similarity in msl.items():
    c1, c2 = concept_pair
    if c1 in embeddings and c2 in embeddings:
        emb1 = embeddings[c1]
        emb2 = embeddings[c2]
        pred_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        msl_similarities.append(similarity)
        pred_similarities.append(pred_similarity)
        path_lengths.append(pl_dict[(c1, c2)])

msl_similarities = np.array(msl_similarities)
pred_similarities = np.array(pred_similarities)
path_lengths = np.array(path_lengths)

print("EMBEDDINGS:")
print("Spearman:", spearmanr(msl_similarities, pred_similarities))
print("Pearson:", pearsonr(msl_similarities, pred_similarities))
print(msl_similarities.shape)

# trim NaN's
msl_similarities_slice = msl_similarities[~np.isnan(path_lengths)]
path_lengths = path_lengths[~np.isnan(path_lengths)]

print("\nPATH LENGTHS:")
print("Spearman:", spearmanr(msl_similarities_slice, path_lengths))
print("Pearson:", pearsonr(msl_similarities_slice, path_lengths))
print(msl_similarities_slice.shape)

# linear regression
model = LinearRegression()
model.fit(msl_similarities.reshape(-1, 1), pred_similarities)
y_emb = model.predict(msl_similarities.reshape(-1, 1))

# again, for path lengths
model = LinearRegression()
model.fit(msl_similarities_slice.reshape(-1, 1), path_lengths)
y_pl = model.predict(msl_similarities_slice.reshape(-1, 1))

plt.scatter(msl_similarities, pred_similarities)
plt.plot(msl_similarities, y_emb, color='red')
plt.xlabel("Multi-SimLex")
plt.ylabel("embedding similarity")
plt.title("SDNE")
# plt.savefig("figures/SDNE.pdf")
plt.show()
plt.cla()


plt.scatter(msl_similarities_slice, path_lengths)
plt.plot(msl_similarities_slice, y_pl, color='red')
plt.xlabel("Multi-SimLex")
plt.ylabel("shortest walking distance")
plt.title("Baseline")
# plt.savefig("figures/Baseline.pdf")
plt.show()


# sys.exit(0)
###############################################

# example words
words = ["ARM", "HAND", "LEG", "FOOT", "FIRE", "WOOD", "TREE", "BOY", "GIRL", "SON", "DAUGHTER"]
# words = ["GIRL", "BOY", "WOMAN", "MAN"]
# words = ["BALD", "NAKED", "CROW", "EAGLE", "VULTURE", "HAWK"]
# words = [c.concepticon_gloss for c in Concepticon().conceptlists['Swadesh-1964-100'].concepts.values()
  #       if c.concepticon_gloss in embeddings]

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
# plt.title("TSNE")
plt.savefig("figures/TSNE-clics4.pdf")
plt.show()
