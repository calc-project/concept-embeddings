import numpy as np


def graph_to_undirected(graph: np.array):
    for i in range(len(graph)):
        for j in range(i):
            cell_value = graph[i, j] + graph[j, i]
            graph[i, j] = graph[j, i] = cell_value

    return graph


def merge_graphs(graph1: np.array, graph2: np.array, concept_to_id1: dict, concept_to_id2: dict, directed: bool = False):
    # generate combined id dict
    merged_concept_to_id = concept_to_id1.copy()
    for concept in concept_to_id2:
        if concept not in merged_concept_to_id:
            merged_concept_to_id[concept] = len(merged_concept_to_id)

    # extend graph1 by n rows and columns, where n is the number of nodes that are only found in graph2
    padding_width = len(merged_concept_to_id) - len(concept_to_id1)
    merged_graph = np.pad(graph1, (0, padding_width))

    # revert concept dict for graph2
    id_to_concept2 = {i: c for c, i in concept_to_id2.items()}

    for i in range(len(graph2)):
        for j in range(len(graph2)):
            if i < j or directed:
                cell = graph2[i, j]
                new_i = merged_concept_to_id[id_to_concept2[i]]
                new_j = merged_concept_to_id[id_to_concept2[j]]
                merged_graph[new_i, new_j] += cell
                if not directed:
                    merged_graph[new_j, new_i] += cell

    return merged_graph, merged_concept_to_id
