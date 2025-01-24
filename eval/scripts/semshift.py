import matplotlib.pyplot as plt
import numpy as np
import seaborn
import csv
import random
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

from graphembeddings.utils.io import read_embeddings, read_ft_embeddings, read_graph_data
from graphembeddings.utils.postprocess import fuse_embeddings, cosine_similarity


GRAPH_EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings" / "babyclics"
FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"

SHIFTS_DEFAULT_FP = Path(__file__).parent.parent / "data" / "semshift" / "shift_summary.tsv"


def load_embeddings(fp, fasttext=False):
    if fasttext:
        return read_ft_embeddings(fp)
    else:
        return read_embeddings(fp)


def load_shifts(fp=SHIFTS_DEFAULT_FP, embeddings=None):
    shifts = []

    header = True
    with open(fp) as f:
        for line in f:
            if header:
                header = False
                continue
            fields = line.strip().split("\t")
            source, target = fields[0], fields[1]
            if embeddings and source in embeddings and target in embeddings:
                shifts.append((fields[0], fields[1]))

    return shifts


def sample_random_shifts(shifts, concepts):
    valid_shifts = []  # shifts where both concepts are found in `concepts` (i.e. have an embedding)
    random_shifts = []

    for c1, c2 in shifts:
        if c1 in concepts and c2 in concepts:
            valid_shifts.append((c1, c2))
            # randomly replace one of the two concepts
            random_concept = random.choice(concepts)
            random_shifts.append((random_concept, random.choice([c1, c2])))

    return valid_shifts, random_shifts


def generate_training_data(true_shifts, random_shifts, embeddings):
    X, y = [], []

    for shift in true_shifts:
        c1, c2 = shift
        if not (c1 in embeddings and c2 in embeddings):
            continue
        sim = cosine_similarity(embeddings[c1], embeddings[c2])
        X.append(sim)
        y.append(1)

    for shift in random_shifts:
        c1, c2 = shift
        if not (c1 in embeddings and c2 in embeddings):
            continue
        sim = cosine_similarity(embeddings[c1], embeddings[c2])
        X.append(sim)
        y.append(0)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y


def fit_logistic_regression(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)

    return lr


if __name__ == "__main__":
    models = ["n2v-cbow", "n2v-sg", "sdne", "prone"]
    headers = ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]

    ft_langs = ["arabic", "english", "spanish", "estonian", "finnish", "french", "polish", "russian", "chinese"]

    # load concepts from the full colexification graph; they are a subset of the concept spaces of all other models
    fullfams_graph_fp = Path(__file__).parent.parent.parent / "data" / "graphs" / "babyclics" / "fullfams.json"
    _, _, concepts = read_graph_data(fullfams_graph_fp)
    shared_concepts = list(concepts.keys())

    # sample shifts and random shifts w.r.t. the concepts in the full colex graph.
    # all models are evaulated on the same sample to ensure a fair comparison
    shifts = load_shifts(embeddings=shared_concepts)

    tables = []
    ft_acc_tables = []

    # sample random shifts 10 times:
    for _ in range(10):
        shifts, random_shifts = sample_random_shifts(shifts, shared_concepts)
        table = []

        for model in models:
            # row in the resulting table
            accuracies = []

            # load individual embeddings
            full_embeddings = load_embeddings(GRAPH_EMBEDDINGS_DIR / "fullfams" / f"{model}.json")
            affix_embeddings = load_embeddings(GRAPH_EMBEDDINGS_DIR / "affixfams" / f"{model}.json")
            overlap_embeddings = load_embeddings(GRAPH_EMBEDDINGS_DIR / "overlapfams" / f"{model}.json")

            # fuse embeddings
            embeddings_full_affix = fuse_embeddings(full_embeddings, affix_embeddings)
            embeddings_full_overlap = fuse_embeddings(full_embeddings, overlap_embeddings)
            embeddings_all = fuse_embeddings(full_embeddings, affix_embeddings, overlap_embeddings)

            for emb_model in [full_embeddings, affix_embeddings, overlap_embeddings,
                               embeddings_full_affix, embeddings_full_overlap, embeddings_all]:
                X, y = generate_training_data(shifts, random_shifts, emb_model)
                lr = fit_logistic_regression(X, y)
                accuracies.append(lr.score(X, y))

            table.append(accuracies)

        # append local table to global "table"
        tables.append(table)

        # check against monolingual fasttext embeddings
        ft_accuracies = []
        ft_headers = []

        for lang in ft_langs:
            embeddings = load_embeddings(FT_EMBEDDINGS_DIR / f"{lang}_embeddings.csv", fasttext=True)
            X, y = generate_training_data(shifts, random_shifts, embeddings)
            lr = fit_logistic_regression(X, y)
            ft_accuracies.append(lr.score(X, y))

        ft_acc_tables.append([ft_accuracies])

    # average over all obtained accuracies per cell
    emb_table = np.mean(tables, axis=0)
    print(tabulate(emb_table, headers=headers, showindex=models, tablefmt="github", floatfmt=".4f"))
    print(100 * "=")

    # same for the fasttext accuracies
    ft_table = np.mean(ft_acc_tables, axis=0)
    print(tabulate(ft_table, headers=ft_langs, tablefmt="github", floatfmt=".4f"))
