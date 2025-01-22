import csv
import numpy as np
import networkx as nx
from graphembeddings.utils.io import read_graph_data, read_embeddings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from tabulate import tabulate


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


def concatenate_embeddings(embeddings1, embeddings2):
    shared_keys = set(embeddings1.keys()) & set(embeddings2.keys())

    return {k: embeddings1[k] + embeddings2[k] for k in shared_keys}


def fuse_embeddings(embeddings1, embeddings2, n_components=128, retain_all=False):
    concat_embeddings = concatenate_embeddings(embeddings1, embeddings2)
    keys, embeddings = zip(*concat_embeddings.items())
    embeddings = np.array(embeddings)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)

    embeddings = dict(zip(keys, embeddings))

    if retain_all:
        for k, v in embeddings1.items():
            if k not in embeddings and len(v) == n_components:
                embeddings[k] = v
        for k, v in embeddings2.items():
            if k not in embeddings and len(v) == n_components:
                embeddings[k] = v

    return embeddings


if __name__ == "__main__":
    mode = "affixfams"
    models = ["n2v-cbow", "n2v-sg", "sdne", "prone"]

    table = []

    # correlate MSL with shortest path lengths (affix colex only)
    G_affix, _, concept_to_id_affix = read_graph_data(GRAPHS_DIR / "babyclics" / f"{mode}.json", directed=True, to_undirected=True)
    msl = read_msl_data()
    corr_affix = msl_correlation_baseline(msl, G_affix, concept_to_id_affix)

    # ...with full colex only
    G_full, id_to_concept_full, concept_to_id_full = read_graph_data(GRAPHS_DIR / "babyclics" / "fullfams.json", directed=True,
                                                to_undirected=True)
    corr_full = msl_correlation_baseline(msl, G_full, concept_to_id_full)

    # ...and with both
    # TODO make this into a proper method later --
    # this is a dirty hack that just works for now, knowing that the nodes in "fullfams" are a subset of the nodes in "affixfams"
    G_combined = G_affix.copy()
    for i in range(len(G_full)):
        for j in range(i):
            cell = G_full[i, j]
            i_aff = concept_to_id_affix[id_to_concept_full[i]]
            j_aff = concept_to_id_affix[id_to_concept_full[j]]
            G_combined[i_aff, j_aff] += cell
            G_combined[j_aff, i_aff] += cell

    corr_combined = msl_correlation_baseline(msl, G_combined, concept_to_id_affix)

    table.append([corr_full, corr_affix, corr_combined])
    headers = ["full", "affix", "combined"]

    for model in models:
        full_embeddings = read_embeddings(Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "fullfams" / f"{model}.json")
        embeddings = read_embeddings(Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / mode / f"{model}.json")
        # concat_embeddings = concatenate_embeddings(embeddings, full_embeddings)
        # corr = msl_correlation(msl, concat_embeddings)
        fused_embeddings = fuse_embeddings(full_embeddings, embeddings, retain_all=False)
        corr_combined = msl_correlation(msl, fused_embeddings)

        full_embeddings = read_embeddings(
            Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "fullfams" / f"{model}.json")
        corr_full = msl_correlation(msl, full_embeddings)

        affix_embeddings = read_embeddings(
            Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "affixfams" / f"{model}.json")
        corr_affix = msl_correlation(msl, affix_embeddings)

        table.append([corr_full, corr_affix, corr_combined])

    index = ["baseline"] + models
    print(tabulate(table, headers=headers, showindex=index, tablefmt="github", floatfmt=".4f"))
