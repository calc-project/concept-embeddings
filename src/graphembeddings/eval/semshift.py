import numpy as np
import random
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from graphembeddings.utils.io import read_embeddings, read_ft_embeddings


GRAPH_EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings"
FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"

SHIFTS_DEFAULT_FP = Path(__file__).parent.parent / "data" / "semshift" / "shift_summary.tsv"


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


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
        if np.isnan(sim):
            continue
        X.append(sim)
        y.append(1)

    for shift in random_shifts:
        c1, c2 = shift
        if not (c1 in embeddings and c2 in embeddings):
            continue
        sim = cosine_similarity(embeddings[c1], embeddings[c2])
        if np.isnan(sim):
            continue
        X.append(sim)
        y.append(0)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y


def generate_baseline_training_data(true_shifts, random_shifts, similarity_function):
    X, y = [], []

    for (true, random) in zip(true_shifts, random_shifts):
        true_sim = similarity_function(*true)
        random_sim = similarity_function(*random)

        if true_sim is np.nan or random_sim is np.nan:
            continue

        X.append(true_sim)
        y.append(1)
        X.append(random_sim)
        y.append(0)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y


def fit_logistic_regression(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)

    return lr
