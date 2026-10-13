import numpy as np
import json

from conceptembeddings.utils.graphutils import graph_to_undirected


def read_graph_data(fp, directed=False, to_undirected=False):
    with open(fp) as f:
        data = json.load(f)

    concept_to_id = data["concept_ids"]
    id_to_concept = {i: c for c, i in concept_to_id.items()}
    edgelist = data["edgelist"]
    concept_coverage = data.get("concept_coverage")

    graph = np.zeros((len(concept_to_id), len(concept_to_id)))
    for i, j, w in edgelist:
        graph[i, j] = w
        if not directed:
            graph[j, i] = w

    if directed and to_undirected:
        graph = graph_to_undirected(graph)

    return graph, id_to_concept, concept_to_id, concept_coverage


def read_embeddings(fp, metadata=False):
    with open(fp) as f:
        data = json.load(f)

    if not metadata:
        return data["embeddings"]

    return data
