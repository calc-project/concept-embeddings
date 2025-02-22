import numpy as np
from pathlib import Path
from tabulate import tabulate
from pynorare import NoRaRe

from graphembeddings.utils.io import read_embeddings

from semshift import load_embeddings, sample_random_shifts, generate_training_data, generate_baseline_training_data, fit_logistic_regression
from baselines import Baseline, get_all_graphs


GRAPH_EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings"
FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"

NORARE_DEFAULT_FP = Path(__file__).parent.parent / "data" / "norare" / "norare-data"


def load_eat_edges(fp=NORARE_DEFAULT_FP, threshold=5):
    norare = NoRaRe(fp)
    eat = norare.datasets.get("Kiss-1973-EAT")

    edges, weights = [], []
    visited = set()
    overflow = ""

    for row in eat.concepts.values():
        c1 = row["concepticon_gloss"]
        for edge in row.get("edges", []):
            # this is necessary for handling the apostrophe, which is represented by its hex code
            if not ":" in edge:
                overflow = edge.replace("&#39", "'")
                continue
            c2, weight = edge.split(":")
            if overflow:
                c2 = overflow + c2
            overflow = ""
            weight = int(weight)
            if weight > threshold and c2 not in visited:
                edges.append((c1, c2))
                weights.append(weight)
        visited.add(c1)

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

    # sample random shifts 50 times:
    for _ in range(50):
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
    print("## Embeddings & Baselines")
    print(tabulate(emb_table, headers=headers, showindex=index, tablefmt="github", floatfmt=".4f"))

    # same for the fasttext accuracies
    ft_table = np.mean(ft_acc_tables, axis=0)
    print("\n## FastText")
    ft_mean_acc = np.mean(ft_table)
    ft_table = [ft_table[0].tolist()]
    ft_table[0].append(ft_mean_acc)
    ft_langs.append("mean")
    print(tabulate(ft_table, headers=ft_langs, tablefmt="github", floatfmt=".4f"))
