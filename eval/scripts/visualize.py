import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate
from newick import loads
from pylocluster import linkage
from pathlib import Path
from pyconcepticon import Concepticon
from adjustText import adjust_text

from graphembeddings.utils.io import read_embeddings

from multisimlex import read_msl_data


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


def generic_plot(concepts, res, title, save_fp=None):
    plt.cla()
    plt.scatter(*np.swapaxes(res, 0, 1), s=15)
    # for concept, coordinates in zip(concepts, res):
    #    plt.annotate(concept, coordinates)
    plt.title(title)
    labels = [plt.text(x, y, concept, ha="center", va="center", size=9) for (x, y), concept in zip(res, concepts)]
    adjust_text(labels, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    if save_fp:
        plt.savefig(save_fp)
    else:
        plt.show()


def pca_plot(concepts, embeddings, save_fp=None):
    matrix = np.array([embeddings[c] for c in concepts])
    pca = PCA(n_components=2)
    res = pca.fit_transform(matrix)
    generic_plot(concepts, res, "PCA", save_fp=save_fp)


def tsne_plot(concepts, embeddings, perplexity=2, save_fp=None, title="TSNE"):
    matrix = np.array([embeddings[c] for c in concepts])
    tsne = TSNE(n_components=2, perplexity=perplexity)
    res = tsne.fit_transform(matrix)
    generic_plot(concepts, res, title, save_fp=save_fp)


def common_concepts(conceptlist1, conceptlist2):
    con = Concepticon()

    if conceptlist1 not in con.conceptlists:
        raise KeyError(f"No such concept list in Concepticon: {conceptlist1}")

    if conceptlist2 not in con.conceptlists:
        raise KeyError(f"No such concept list in Concepticon: {conceptlist2}")

    return ({x.concepticon_gloss for x in con.conceptlists[conceptlist1].concepts.values()} &
            {x.concepticon_gloss for x in con.conceptlists[conceptlist2].concepts.values()})


def msl_similarity_matrix(words):
    msl = read_msl_data()

    # set up similarity matrix for Multi-SimLex
    matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:
                if (word1, word2) in msl:
                    matrix[i, j] = matrix[j, i] = msl[(word1, word2)]
                elif (word2, word1) in msl:
                    matrix[i, j] = matrix[j, i] = msl[(word2, word1)]

    # create a dictionary mapping concepts to rows in similarity matrix
    row_dict = {}
    selected_words = []
    for i, word in enumerate(words):
        row = matrix[i]
        if not all(row == 0):
            row_dict[word] = matrix[i]
            selected_words.append(word)

    return row_dict, selected_words


if __name__ == "__main__":
    EMB_DIR = Path(__file__).parent.parent.parent / "embeddings" / "babyclics"
    OUT_DIR = Path(__file__).parent.parent / "figures"

    words = common_concepts("Vulic-2020-2244", "Swadesh-1964-100")

    row_dict, selected_words = msl_similarity_matrix(words)
    tsne_plot(selected_words, row_dict, perplexity=4, title="Multi-SimLex", save_fp=OUT_DIR / "msl.pdf")

    for mode in ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]:
        if "+" not in mode:
            dir_name = EMB_DIR / f"{mode}fams"
        else:
            dir_name = EMB_DIR / mode.replace("+", "-")

        embeddings = read_embeddings(dir_name / "prone.json")
        valid_words = words & set(embeddings.keys())
        tsne_plot(valid_words, embeddings, perplexity=4, title=mode, save_fp=OUT_DIR / f"{mode.replace("+", "-")}.pdf")
