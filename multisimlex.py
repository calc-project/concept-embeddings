import csv
import networkx as nx
from pyconcepticon import Concepticon


cnc = Concepticon() # add path to concepticon if you did not run `cldfbench catconfig`
# get the concept list by
cl = {
        concept.number: concept for concept in cnc.conceptlists[
                "Vulic-2020-2244"].concepts.values()}

# this little dictionary will handle keys to all data in Multi-Simlex
msl = {}
for concept in cl.values():
    for (idx, link, eng, rus, chin, can, ara, span, pol, fra, est, fin) in zip(
            concept.attributes["simlex_ids"],
            concept.attributes["links"],
            concept.attributes["english_score"],
            concept.attributes["russian_score"],
            concept.attributes["chinese_score"],
            concept.attributes["cantonese_score"],
            concept.attributes["arabic_score"],
            concept.attributes["spanish_score"],
            concept.attributes["polish_score"],
            concept.attributes["french_score"],
            concept.attributes["estonian_score"],
            concept.attributes["finnish_score"]
        ):
        msl[idx] = [
                concept.concepticon_id or "",
                concept.concepticon_gloss or "",
                eng, rus, chin, can, ara, span, pol, fra, est, fin
                ]

# enrich by shortest path length information
with open("clics-edgelist.tsv") as f:
    edges = f.read().split("\n")

graph = nx.parse_edgelist(edges, nodetype=int, data=(("weight", int),))

# inverse weights (higher weights = stronger connection => shorter path)
for x, y in graph.edges:
    graph[x][y]["inv_weight"] = 1 / graph[x][y]["weight"]

# load concept to idx
id_dict = {}

with open("clics-concept-ids.tsv") as f:
    for row in f.read().split("\n"):
        if not row:
            continue
        idx, concept = row.split("\t")
        id_dict[concept] = int(idx)

# write multisimlex data
with open("multisimlex.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["CID1", "CGL1", "CID2", "CGL2", "eng", "rus", "cmn", "yue", "ara",
                     "spa", "pol", "fra", "est", "fin", "mean", "shortest_path_weighted", "shortest_path_unweighted"])
    for i in range(len(msl) // 2):
        row1 = msl[f"{i+1}:1"]
        row2 = msl[f"{i+1}:2"]
        if not (row1[0] and row2[0]):
            continue
        # get shortest path between concepts
        c1 = row1[1]
        c2 = row2[1]
        i1 = id_dict.get(c1)
        i2 = id_dict.get(c2)

        if i1 and i2 and nx.has_path(graph, i1, i2):
            pathlength_weighted = nx.shortest_path_length(graph, i1, i2, "inv_weight")
            pathlength_unweighted = nx.shortest_path_length(graph, i1, i2)
        else:
            pathlength_weighted = pathlength_unweighted = ""

        mean = sum(row1[2:]) / len(row1[2:])
        combined = row1[:2] + row2 + [mean, pathlength_weighted, pathlength_unweighted]
        # combined = row1[:2] + row2 + [mean]
        writer.writerow(combined)
