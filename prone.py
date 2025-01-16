import networkx as nx
from nodevectors import ProNE


graph = nx.Graph()

#with open("clics4/edgelist-Family_Weight.tsv") as f:
with open("clics-edgelist.tsv") as f:
    for line in f:
        if not line or line.startswith("#"):
            continue
        source, target, weight = line.strip().split("\t")
        graph.add_edge(source, target, weight=float(weight))

model = ProNE(n_components=128)
model.fit(graph)

id_to_concept = {}

with open("clics-concept-ids.tsv") as f:
    for line in f:
        if not line:
            continue
        id, concept = line.strip().split("\t")
        id_to_concept[id] = concept

with open("embeddings/2025-01-16/prone-128-babyclics.tsv", "w") as f:
    for id, emb in model.model.items():
        concept = id_to_concept[id]
        f.write(f"{concept}\t{emb.tolist()}\n")
