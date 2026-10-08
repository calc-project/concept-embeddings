import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate
from pathlib import Path
from pyconcepticon import Concepticon
from adjustText import adjust_text

from conceptembeddings.pretrained import ConceptEmbeddings


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


def generic_plot(concepts, res, title, save_fp=None, highlight=None):
    plt.cla()
    if highlight:
        colors = len(concepts) * ["b"]
        for c in highlight:
            if c in concepts:
                colors[concepts.index(c)] = "r"
        plt.scatter(*np.swapaxes(res, 0, 1), s=15, c=colors)
    else:
        plt.scatter(*np.swapaxes(res, 0, 1), s=15)
    plt.title(title)
    if highlight:
        labels = [plt.text(x, y, concept, ha="center", va="center", size=9, color="r" if concept in highlight else "k")
              for (x, y), concept in zip(res, concepts)]
    else:
        labels = [plt.text(x, y, concept, ha="center", va="center", size=9)
                  for (x, y), concept in zip(res, concepts)]
    adjust_text(labels, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    plt.xticks([])
    plt.yticks([])
    if save_fp:
        plt.savefig(save_fp)
    else:
        plt.show()


def pca_plot(concepts, embeddings, save_fp=None, title="PCA", highlight=None):
    concepts = list(concepts)
    matrix = np.array([embeddings[c] for c in concepts])
    pca = PCA(n_components=2)
    res = pca.fit_transform(matrix)
    if highlight:
        suffix = save_fp.suffix if save_fp else None
        fp = Path(str(save_fp).replace(suffix, f"-hl{suffix}")) if save_fp else None
        generic_plot(concepts, res, title, save_fp=fp, highlight=highlight)
    generic_plot(concepts, res, "PCA", save_fp=save_fp)


def tsne_plot(concepts, embeddings, perplexity=2, save_fp=None, title="TSNE", highlight=None):
    concepts = list(concepts)
    matrix = np.array([embeddings[c] for c in concepts])
    tsne = TSNE(n_components=2, perplexity=perplexity)
    res = tsne.fit_transform(matrix)
    if highlight:
        suffix = save_fp.suffix if save_fp else None
        fp = Path(str(save_fp).replace(suffix, f"-hl{suffix}")) if save_fp else None
        generic_plot(concepts, res, title, save_fp=fp, highlight=highlight)
    generic_plot(concepts, res, title, save_fp=save_fp)


if __name__ == "__main__":
    OUT_DIR = Path(__file__).parent / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    words = {c.concepticon_gloss for c in Concepticon().conceptlists["Holman-2008-40"].concepts.values()}
    embedder = ConceptEmbeddings()
    embeddings = embedder.generate_embeddings(words)
    test_word = "ORC"
    embeddings[test_word] = embedder.embed_definition("A fictional, aggressive humanoid creature common in fantasy literature and games.")
    words.add(test_word)
    tsne_plot(words, embeddings, perplexity=5, highlight=[test_word], save_fp=OUT_DIR / "orc-tsne.pdf", title="")
    pca_plot(words, embeddings, highlight=[test_word], save_fp=OUT_DIR / "orc-pca.pdf", title="")
