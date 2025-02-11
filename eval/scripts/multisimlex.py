import csv
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tabulate import tabulate

from graphembeddings.utils.io import read_embeddings, read_ft_embeddings

from baselines import Baseline, get_all_graphs


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

    # print(f"Correlation with Multi-SimLex calculated based on {len(msl_similarities)}/{len(similarity_ratings)} "
    #       f"available concept pairs.")

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

    # print(f"Correlation with Multi-SimLex calculated based on {len(msl_similarities)}/{len(similarity_ratings)} "
    #       f"available concept pairs.")

    if correlation_measure == "spearman":
        corr = spearmanr(msl_similarities, path_lengths)
    elif correlation_measure == "pearson":
        corr = pearsonr(msl_similarities, path_lengths)
    else:
        raise ValueError(f"Correlation measure {correlation_measure} not recognized. Available options: \"spearman\", \"pearson\".")

    return corr.statistic


if __name__ == "__main__":
    baseline_models = ["shortest path", "cosine sim", "ppmi", "random walks"]
    models = ["n2v-cbow", "n2v-sg", "sdne", "prone"]
    headers = ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]

    table = []

    # read Multi-SimLex data
    msl = read_msl_data()

    graphs, concept_ids = get_all_graphs()

    for h in headers:
        line = []

        # baselines
        graph = graphs[h]
        concept_to_id = concept_ids[h]
        baseline = Baseline(graph, concept_to_id)

        walk_lengths, cos_sims, ppmis, random_walks_sim = [], [], [], []
        for pair, value in msl.items():
            walk_lengths.append(baseline.shortest_path(*pair))
            cos_sims.append(baseline.cos_similarity(*pair))
            ppmis.append(baseline.ppmi(*pair))
            random_walks_sim.append(baseline.katz_random_walks(*pair))

        msl_sims = list(msl.values())
        line.append(spearmanr(walk_lengths, msl_sims, nan_policy="omit").statistic)
        line.append(spearmanr(cos_sims, msl_sims, nan_policy="omit").statistic)
        line.append(spearmanr(ppmis, msl_sims, nan_policy="omit").statistic)
        line.append(spearmanr(random_walks_sim, msl_sims, nan_policy="omit").statistic)

        # embeddings
        if "+" in h:
            name = h.replace("+", "-")
        else:
            name = h + "fams"

        for model in models:
            embeddings = read_embeddings(EMBEDDINGS_DIR / name / f"{model}.json")
            corr = msl_correlation(msl, embeddings)
            line.append(corr)

        table.append(line)

    table = np.array(table).swapaxes(0, 1).tolist()

    index = baseline_models + models
    print("## Embeddings & Baselines")
    print(tabulate(table, headers=headers, showindex=index, tablefmt="github", floatfmt=".4f"))

    # evaluate on fasttext as baseline
    ft_table = []
    ft_langs = ["arabic", "english", "spanish", "estonian", "finnish", "french", "polish", "russian", "chinese"]

    for lang in ft_langs:
        embeddings = read_ft_embeddings(Path(__file__).parent.parent / "data" / "fasttext" / f"{lang}_embeddings.csv")
        embeddings_filtered = {k: v for k, v in embeddings.items() if k in concept_ids["full+affix+overlap"]}
        corr = msl_correlation(msl, embeddings)
        corr_filtered = msl_correlation(msl, embeddings_filtered)
        ft_table.append([corr, corr_filtered])

    mean, mean_filtered = np.mean(ft_table, axis=0)
    ft_table.append([mean, mean_filtered])
    ft_langs.append("mean")
    print("\n## FastText")
    print(tabulate(ft_table, headers=["all", "filtered"], showindex=ft_langs, tablefmt="github", floatfmt=".4f"))
