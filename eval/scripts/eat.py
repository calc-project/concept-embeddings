import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tabulate import tabulate
from sklearn.linear_model import LinearRegression

from graphembeddings.utils.io import read_embeddings, read_graph_data
from graphembeddings.utils.postprocess import cosine_similarity, fuse_embeddings

from semshift import load_embeddings, sample_random_shifts, generate_training_data, generate_baseline_training_data, fit_logistic_regression
from baselines import Baseline, get_all_graphs


GRAPH_EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings" / "babyclics"
FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"

EAT_DEFAULT_FP = Path(__file__).parent.parent / "data" / "eat" / "Kiss-1973-EAT.tsv"


def load_eat_edges(fp=EAT_DEFAULT_FP, threshold=5):
    edges = []
    weights = []

    with open(fp) as f:
        visited = set()

        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            concept = row["CONCEPTICON_GLOSS"]
            edge_data = row["EDGES"]
            for edge in edge_data.split(";"):
                # that happens for UC-dec encoding of some characters, e.g. &#39; for the apostrophe
                if not (edge and ":" in edge):
                    continue
                target_concept, weight = edge.split(":")
                if target_concept in visited:
                    continue
                weight = int(weight)
                if weight > threshold:
                    edges.append((concept, target_concept))
                    weights.append(weight)
            visited.add(concept)

    return edges, weights


if __name__ == "__main__":
    baseline_models = ["shortest path", "cosine sim", "ppmi", "random walks"]
    models = ["n2v-cbow", "n2v-sg", "sdne", "prone"]
    headers = ["full", "affix", "overlap", "full+affix", "full+overlap", "full+affix+overlap"]

    ft_langs = ["arabic", "english", "spanish", "estonian", "finnish", "french", "polish", "russian", "chinese"]

    # load concepts from the full colexification graph; they are a subset of the concept spaces of all other models
    graphs, concept_ids = get_all_graphs()
    shared_concepts = list(concept_ids["full+affix+overlap"].keys())

    # load edges from EAT
    edges, weights = load_eat_edges()

    # accuracies for logistic regression (link prediction)
    tables = []
    ft_acc_tables = []

    # sample random shifts 10 times:
    for _ in range(10):
        true_edges, random_edges = sample_random_shifts(edges, shared_concepts)
        table = []

        for h in headers:
            accuracies = []

            # baselines
            graph = graphs[h]
            concept_to_id = concept_ids[h]
            baseline = Baseline(graph, concept_to_id)

            for baseline_model in baseline_models:
                if baseline_model == "shortest path":
                    X, y = generate_baseline_training_data(true_edges, random_edges, similarity_function=baseline.shortest_path)
                elif baseline_model == "cosine sim":
                    X, y = generate_baseline_training_data(true_edges, random_edges, similarity_function=baseline.cos_similarity)
                elif baseline_model == "ppmi":
                    X, y = generate_baseline_training_data(true_edges, random_edges, similarity_function=baseline.ppmi)
                elif baseline_model == "random walks":
                    X, y = generate_baseline_training_data(true_edges, random_edges, similarity_function=baseline.katz_random_walks)
                else:
                    continue

                lr = fit_logistic_regression(X, y)
                accuracies.append(lr.score(X, y))

            # models
            if "+" in h:
                name = h.replace("+", "-")
            else:
                name = h + "fams"

            for model in models:
                embeddings = read_embeddings(GRAPH_EMBEDDINGS_DIR / name / f"{model}.json")
                X, y = generate_training_data(true_edges, random_edges, embeddings)
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
            X, y = generate_training_data(true_edges, random_edges, embeddings)
            lr = fit_logistic_regression(X, y)
            ft_accuracies.append(lr.score(X, y))

        ft_acc_tables.append([ft_accuracies])

    # average over all obtained accuracies per cell
    emb_table = np.mean(tables, axis=0)
    emb_table = emb_table.swapaxes(0, 1).tolist()
    index = baseline_models + models
    print(tabulate(emb_table, headers=headers, showindex=index, tablefmt="github", floatfmt=".4f"))
    print(200 * "-")

    # same for the fasttext accuracies
    ft_table = np.mean(ft_acc_tables, axis=0)
    print(tabulate(ft_table, headers=ft_langs, tablefmt="github", floatfmt=".4f"))

    print(200 * "=")

    ### LINEAR REGRESSION ON ATTESTED LINKS
    LOG_TRANSFORM = True
    table = []

    for model in models:
        line = []
        for h in headers:
            if "+" in h:
                name = h.replace("+", "-")
            else:
                name = h + "fams"
            embeddings = read_embeddings(GRAPH_EMBEDDINGS_DIR / name / f"{model}.json")
            X, y = [], []
            for pair, weight in zip(edges, weights):
                c1, c2 = pair
                if c1 in embeddings and c2 in embeddings:
                    X.append(cosine_similarity(embeddings[c1], embeddings[c2]))
                    y.append(weight)
            X = np.array(X).reshape(-1, 1)
            y = np.array(y)
            if LOG_TRANSFORM:
                y = np.log(np.log(y))
            linear = LinearRegression()
            linear.fit(X, y)
            line.append(linear.score(X, y))
        table.append(line)

    print(tabulate(table, headers=headers, showindex=models, tablefmt="github", floatfmt=".4f"))
    print(200 * "-")

    ft_table = []

    for lang in ft_langs:
        embeddings = load_embeddings(FT_EMBEDDINGS_DIR / f"{lang}_embeddings.csv", fasttext=True)
        X, y = [], []
        for pair, weight in zip(edges, weights):
            c1, c2 = pair
            if c1 in embeddings and c2 in embeddings:
                X.append(cosine_similarity(embeddings[c1], embeddings[c2]))
                y.append(weight)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        if LOG_TRANSFORM:
            y = np.log(np.log(y))
        linear = LinearRegression()
        linear.fit(X, y)
        ft_table.append(linear.score(X, y))

    print(tabulate([ft_table], headers=ft_langs, tablefmt="github", floatfmt=".4f"))
