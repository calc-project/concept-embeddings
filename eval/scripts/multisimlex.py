import csv
import numpy as np
import networkx as nx
from graphembeddings.utils.io import read_graph_data, read_embeddings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr


MSL_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "msl" / "multisimlex.csv"
GRAPHS_DIR = Path(__file__).parent.parent.parent / "data" / "graphs"


def read_msl_data(fp=MSL_DEFAULT_PATH, col="mean"):
    similarity_ratings = {}

    with open(fp) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if col not in row:
                raise ValueError(f"Column {col} not found in {fp}")
            c1 = row["CONCEPT_1"]
            c2 = row["CONCEPT_2"]
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

    print(f"Correlation with Multi-SimLex calculated based on {len(msl_similarities)}/{len(similarity_ratings)} "
          f"available concept pairs.")

    if correlation_measure == "spearman":
        corr = spearmanr(msl_similarities, embedding_similarities)
    elif correlation_measure == "pearson":
        corr = pearsonr(msl_similarities, embedding_similarities)
    else:
        raise ValueError(f"Correlation measure {correlation_measure} not recognized. Available options: \"spearman\", \"pearson\".")

    return corr.statistic


def msl_correlation_baseline(similarity_ratings, graph, concept_to_id, correlation_measure="spearman", directed=False):
    """
    Baseline correlation, inferring similarites directly from the graph via shortest paths.
    :return:
    """
    # invert weights of the edges
    graph = np.divide(1, graph, where=graph > 0)
    if directed:
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(graph)

    msl_similarities = []
    path_lengths = []

    for concept_pair, similarity in similarity_ratings.items():
        c1, c2 = concept_pair
        id1 = concept_to_id.get(c1)
        id2 = concept_to_id.get(c2)
        if id1 in graph and id2 in graph and nx.has_path(graph, id1, id2):
            msl_similarities.append(similarity)
            path_lengths.append(nx.shortest_path_length(graph, id1, id2))

    print(f"Correlation with Multi-SimLex calculated based on {len(msl_similarities)}/{len(similarity_ratings)} "
          f"available concept pairs.")

    if correlation_measure == "spearman":
        corr = spearmanr(msl_similarities, path_lengths)
    elif correlation_measure == "pearson":
        corr = pearsonr(msl_similarities, path_lengths)
    else:
        raise ValueError(f"Correlation measure {correlation_measure} not recognized. Available options: \"spearman\", \"pearson\".")

    return corr.statistic


if __name__ == "__main__":
    G, _, concept_to_id = read_graph_data(GRAPHS_DIR / "clics4" / "family_count.json")
    msl = read_msl_data()
    corr = msl_correlation_baseline(msl, G, concept_to_id)
    print(corr)

    embeddings = read_embeddings(Path(__file__).parent.parent.parent / "prone-test.json")
    corr = msl_correlation(msl, embeddings)
    print(corr)
