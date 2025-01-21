import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate
from newick import loads
from pylocluster import linkage
from pathlib import Path

from graphembeddings.utils.io import read_embeddings


def create_distance_matrix(concepts, embeddings, logging=False):
    distance_matrix = np.zeros((len(concepts), len(concepts)))

    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j:
                emb1 = embeddings[c1]
                emb2 = embeddings[c2]
                cos_distance = 1 - (np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                distance_matrix[i, j] = distance_matrix[j, i] = cos_distance

    if logging:
        print(tabulate(distance_matrix, headers=concepts, showindex=concepts, floatfmt=".4f"))

    return distance_matrix


def tree(distance_matrix, concepts):
    print(loads(linkage(distance_matrix, taxa=concepts))[0].ascii_art())


def generic_plot(concepts, res, title):
    plt.cla()
    plt.scatter(*np.swapaxes(res, 0, 1))
    for concept, coordinates in zip(concepts, res):
        plt.annotate(concept, coordinates)
    plt.title(title)
    plt.show()


def pca_plot(concepts, embeddings):
    matrix = np.array([embeddings[c] for c in concepts])
    pca = PCA(n_components=2)
    res = pca.fit_transform(matrix)
    generic_plot(concepts, res, "PCA")


def tsne_plot(concepts, embeddings, perplexity=2):
    matrix = np.array([embeddings[c] for c in concepts])
    tsne = TSNE(n_components=2, perplexity=perplexity)
    res = tsne.fit_transform(matrix)
    generic_plot(concepts, res, "TSNE")


if __name__ == "__main__":
    words = ["ARM", "HAND", "LEG", "FOOT", "FIRE", "WOOD", "TREE", "BOY", "GIRL", "SON", "DAUGHTER"]
    embeddings = read_embeddings(Path(__file__).parent.parent.parent / "prone-test.json")

    distance_matrix = create_distance_matrix(words, embeddings, logging=True)
    tree(distance_matrix, words)
    pca_plot(words, embeddings)
    tsne_plot(words, embeddings, perplexity=3)
