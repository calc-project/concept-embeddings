import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate
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
    # for concept, coordinates in zip(concepts, res):
    #    plt.annotate(concept, coordinates)
    plt.title(title)
    if highlight:
        labels = [plt.text(x, y, concept, ha="center", va="center", size=9, color="r" if concept in highlight else "k")
              for (x, y), concept in zip(res, concepts)]
    else:
        labels = [plt.text(x, y, concept, ha="center", va="center", size=9)
                  for (x, y), concept in zip(res, concepts)]
    adjust_text(labels, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    if save_fp:
        plt.savefig(save_fp)
    else:
        plt.show()


def pca_plot(concepts, embeddings, save_fp=None):
    concepts = list(concepts)
    matrix = np.array([embeddings[c] for c in concepts])
    pca = PCA(n_components=2)
    res = pca.fit_transform(matrix)
    generic_plot(concepts, res, "PCA", save_fp=save_fp)


def tsne_plot(concepts, embeddings, perplexity=2, save_fp=None, title="TSNE", highlight=None):
    concepts = list(concepts)
    matrix = np.array([embeddings[c] for c in concepts])
    tsne = TSNE(n_components=2, perplexity=perplexity)
    res = tsne.fit_transform(matrix)
    if highlight:
        suffix = save_fp.suffix
        fp = Path(str(save_fp).replace(suffix, f"-hl{suffix}"))
        generic_plot(concepts, res, title, save_fp=fp, highlight=highlight)
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

    concept_to_id = {}
    id_to_concept = {}

    for c1, c2 in msl.keys():
        if c1 not in concept_to_id:
            i = len(concept_to_id)
            concept_to_id[c1] = i
            id_to_concept[i] = c1
        if c2 not in concept_to_id:
            i = len(concept_to_id)
            concept_to_id[c2] = i
            id_to_concept[i] = c2

    # set up similarity matrix for Multi-SimLex
    matrix = np.zeros((len(concept_to_id), len(concept_to_id)))
    for (c1, c2), score in msl.items():
        id1, id2 = concept_to_id[c1], concept_to_id[c2]
        matrix[id1, id2] = matrix[id2, id1] = score

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
    EMB_DIR = Path(__file__).parent.parent.parent / "embeddings"
    OUT_DIR = Path(__file__).parent.parent / "figures"

    # retrieve concepts from Swadesh-100 list that are present in all three colexification networks
    words = {c.concepticon_gloss for c in Concepticon().conceptlists["Swadesh-1964-100"].concepts.values()}
    words = words & set(read_embeddings(EMB_DIR / "full-affix-overlap" / "prone.json").keys())

    row_dict, selected_words = msl_similarity_matrix(words)
    tsne_plot(selected_words, row_dict, perplexity=4, title="Multi-SimLex", save_fp=OUT_DIR / "msl.pdf")

    for mode in ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]:
        if "+" not in mode:
            dir_name = EMB_DIR / f"{mode}fams"
        else:
            dir_name = EMB_DIR / mode.replace("+", "-")

        embeddings = read_embeddings(dir_name / "prone.json")
        tsne_plot(words, embeddings, perplexity=4, title=mode, highlight=["BARK", "TREE"],
                  save_fp=OUT_DIR / f"{mode.replace("+", "-")}.pdf")
