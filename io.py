import numpy as np
from collections import defaultdict

from pyconcepticon import Concepticon


def write_network_files(edgelist_file="clics-edgelist.tsv", id_dict_file="clics-concept-ids.tsv"):
    # get the graph
    con = Concepticon()
    clist = con.conceptlists["List-2023-1308"]

    idxs = [c.id for c in clist.concepts.values()]
    concepts = [clist.concepts[idx].concepticon_gloss for idx in idxs]

    # automatically assigns the next id when a concept is first queried
    concept_to_id = defaultdict(lambda: len(concept_to_id))
    edgelist = []
    visited = set()

    for (idx, concept) in zip(idxs, concepts):
        # iterate over all links and fill the matrix
        # skip over unconnected nodes
        if not clist.concepts[idx].attributes["linked_concepts"]:
            continue
        id = concept_to_id[concept]
        for node in clist.concepts[idx].attributes["linked_concepts"]:
            weight = node["FullFams"]
            if weight > 0:
                other_concept = node["NAME"]
                other_id = concept_to_id[other_concept]
                if other_id not in visited:
                    edgelist.append((id, other_id, weight))
        visited.add(id)

    with open(edgelist_file, "w") as f:
        f.write(f"# {len(concept_to_id)}\n")
        for edge in edgelist:
            f.write(f"{edge[0]}\t{edge[1]}\t{edge[2]}\n")

    with open(id_dict_file, "w") as f:
        for concept, id in concept_to_id.items():
            f.write(f"{id}\t{concept}\n")


def read_network_file(edgelist_file="clics-edgelist.tsv"):
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


if __name__ == "__main__":
    write_network_files()
    graph = read_network_file()
    print(graph.ndim, graph.dtype, graph.shape)
