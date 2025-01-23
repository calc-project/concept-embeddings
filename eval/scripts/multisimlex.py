import csv
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tabulate import tabulate

from graphembeddings.utils.io import read_graph_data, read_embeddings
from graphembeddings.utils.graphutils import merge_graphs
from graphembeddings.utils.postprocess import fuse_embeddings


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
    # mode = "affixfams"
    models = ["n2v-cbow", "n2v-sg", "sdne", "prone"]

    table = []

    # read Multi-SimLex data
    msl = read_msl_data()

    # correlate MSL with shortest path lengths (affix colex only)
    G_affix, _, concept_to_id_affix = read_graph_data(GRAPHS_DIR / "babyclics" / "affixfams.json", directed=True, to_undirected=True)
    corr_affix = msl_correlation_baseline(msl, G_affix, concept_to_id_affix)

    # ...with full colex only
    G_full, id_to_concept_full, concept_to_id_full = read_graph_data(GRAPHS_DIR / "babyclics" / "fullfams.json")
    corr_full = msl_correlation_baseline(msl, G_full, concept_to_id_full)

    # ...and with overlap colex.
    G_overlap, id_to_concept_overlap, concept_to_id_overlap = read_graph_data(GRAPHS_DIR / "babyclics" / "overlapfams.json")
    corr_overlap = msl_correlation_baseline(msl, G_overlap, concept_to_id_overlap)

    # ...combine affix and full colex
    G_full_affix, concepts_full_affix = merge_graphs(G_full, G_affix, concept_to_id_full, concept_to_id_affix)
    corr_full_affix = msl_correlation_baseline(msl, G_full_affix, concepts_full_affix)

    # combine overlap and full colex
    G_full_overlap, concepts_full_overlap = merge_graphs(G_full, G_overlap, concept_to_id_full, concept_to_id_overlap)
    corr_full_overlap = msl_correlation_baseline(msl, G_full_overlap, concepts_full_overlap)

    # combine all
    G_all, concepts_all = merge_graphs(G_full_affix, G_overlap, concepts_full_affix, concept_to_id_overlap)
    corr_all = msl_correlation_baseline(msl, G_all, concepts_all)

    table.append([corr_full, corr_affix, corr_overlap, corr_full_affix, corr_full_overlap, corr_all])
    headers = ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]

    for model in models:
        full_embeddings = read_embeddings(Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "fullfams" / f"{model}.json")
        affix_embeddings = read_embeddings(Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "affixfams" / f"{model}.json")
        overlap_embeddings = read_embeddings(Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "overlapfams" / f"{model}.json")

        # calculate correlations for single embeddings
        corr_full = msl_correlation(msl, full_embeddings)
        corr_affix = msl_correlation(msl, affix_embeddings)
        corr_overlap = msl_correlation(msl, overlap_embeddings)

        # fuse embeddings & calculate correlation
        embeddings_full_affix = fuse_embeddings(full_embeddings, affix_embeddings)
        corr_full_affix = msl_correlation(msl, embeddings_full_affix)
        embeddings_full_overlap = fuse_embeddings(full_embeddings, overlap_embeddings)
        corr_full_overlap = msl_correlation(msl, embeddings_full_overlap)
        embeddings_all = fuse_embeddings(full_embeddings, affix_embeddings, overlap_embeddings)
        corr_all = msl_correlation(msl, embeddings_all)

        table.append([corr_full, corr_affix, corr_overlap, corr_full_affix, corr_full_overlap, corr_all])

    index = ["baseline"] + models
    print(tabulate(table, headers=headers, showindex=index, tablefmt="github", floatfmt=".4f"))
