import csv
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr


MSL_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "msl" / "multisimlex.csv"
GRAPHS_DIR = Path(__file__).parent.parent.parent / "data" / "graphs"
EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings"


def read_msl_data(fp=MSL_DEFAULT_PATH, col="mean"):
    similarity_ratings = {}

    with open(fp) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if col not in row:
                raise ValueError(f"Column {col} not found in {fp}")
            c1 = row["CONCEPT_1"]
            c2 = row["CONCEPT_2"]
            if not c1 == c2:
                rating = float(row[col])
                similarity_ratings[(c1, c2)] = rating

    return similarity_ratings


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def msl_correlation(similarity_ratings, embeddings, correlation_measure="spearman"):
    msl_similarities = []
    embedding_similarities = []

    for concept_pair, similarity in similarity_ratings.items():
        c1, c2 = concept_pair
        if c1 in embeddings and c2 in embeddings:
            emb1 = embeddings[c1]
            emb2 = embeddings[c2]
            emb_similarity = cosine_similarity(emb1, emb2)
            msl_similarities.append(similarity)
            embedding_similarities.append(emb_similarity)

    if correlation_measure == "spearman":
        corr = spearmanr(msl_similarities, embedding_similarities, nan_policy="omit")
    elif correlation_measure == "pearson":
        corr = pearsonr(msl_similarities, embedding_similarities)
    else:
        raise ValueError(f"Correlation measure {correlation_measure} not recognized. Available options: \"spearman\", \"pearson\".")

    return corr.statistic
