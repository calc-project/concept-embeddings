import numpy as np
import json
import csv

from graphembeddings.utils.graphutils import graph_to_undirected


def read_network_file(edgelist_file):
    with open(edgelist_file) as f:
        rows = f.read().split("\n")

    assert rows[0].startswith("# ")

    num_nodes = int(rows[0].replace("# ", ""))
    graph = np.zeros((num_nodes, num_nodes))

    for row in rows[1:]:
        if not row:
            continue
        i, j, weight = row.split("\t")
        i = int(i)
        j = int(j)
        weight = float(weight)

        graph[i, j] = weight
        graph[j, i] = weight

    return graph


def read_concept_ids(id_file):
    # load idx to concept
    id_to_concept = {}
    # ...and concept to idx
    concept_to_id = {}

    with open(id_file) as f:
        for row in f.read().split("\n"):
            if not row:
                continue
            idx, concept = row.split("\t")
            id_to_concept[int(idx)] = concept
            concept_to_id[concept] = int(idx)

    return id_to_concept, concept_to_id


def read_graph_data(fp, directed=False, to_undirected=False):
    with open(fp) as f:
        data = json.load(f)

    concept_to_id = data["concept_ids"]
    id_to_concept = {i: c for c, i in concept_to_id.items()}
    edgelist = data["edgelist"]

    graph = np.zeros((len(concept_to_id), len(concept_to_id)))
    for i, j, w in edgelist:
        graph[i, j] = w
        if not directed:
            graph[j, i] = w

    if directed and to_undirected:
        graph = graph_to_undirected(graph)

    return graph, id_to_concept, concept_to_id


def read_embeddings(fp):
    with open(fp) as f:
        data = json.load(f)

    return data["embeddings"]


def read_ft_embeddings(fp):
    embeddings = {}

    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row["CONCEPTICON_GLOSS"]
            emb = eval(row["EMBEDDING"])
            embeddings[concept] = emb

    return embeddings
