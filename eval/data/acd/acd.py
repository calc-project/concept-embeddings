import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from seaborn import kdeplot
from itertools import combinations


with open("cogsets.json") as f:
    cogsets = json.load(f)

cogset_to_color = {
    "*bahi": "red",
    "*bajaq₁": "blue",
    "*quluh": "green",
    "*Sapuy": "black"
}

concepts = []
colors = []

for protoform, cogset in cogsets.items():
    for c in cogset:
        concepts.append(c)
        colors.append(cogset_to_color[protoform])

embeddings = []

with open("embeddings/n2v-cbow.tsv") as f:
    for line in f:
        if not line:
            continue
        concept, emb = line.strip().split("\t")
        if concept not in concepts:
            continue
        emb = np.array(eval(emb))
        embeddings.append(emb)

assert len(concepts) == len(embeddings)

embeddings = np.array(embeddings)
pca = PCA(n_components=2)
res = pca.fit_transform(embeddings)
#tsne = TSNE(n_components=2, perplexity=5)
#res = tsne.fit_transform(embeddings)
#plt.scatter(*np.swapaxes(res, 0, 1), c=colors)
#plt.show()

embeddings_by_cogset = []
tmp = []

for i, emb in enumerate(embeddings):
    if i == 0:
        tmp.append(emb)
    else:
        color = colors[i]
        last_color = colors[i-1]
        if color == last_color:
            tmp.append(emb)
        else:
            embeddings_by_cogset.append(tmp)
            tmp = [emb]

def cos_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / np.linalg.norm(emb1) * np.linalg.norm(emb2)

all_similarities = [cos_similarity(x, y) for x, y in combinations(embeddings, 2)]
cluster_similarities = []
for cluster in embeddings_by_cogset:
    for x, y in combinations(cluster, 2):
        cluster_similarities.append(cos_similarity(x, y))

plt.cla()
kdeplot(all_similarities)
kdeplot(cluster_similarities)
plt.show()

print(np.mean(all_similarities))
print(np.mean(cluster_similarities))
