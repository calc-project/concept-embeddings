from graphembeddings.models.trainer import Node2Vec, ProNE, SDNE
from pathlib import Path


GRAPH_DIR = Path(__file__).parent / "data" / "graphs"
OUTPUT_BASE_DIR = Path(__file__).parent / "embeddings"

available_graphs = {
    "babyclics": ["affixfams", "fullfams", "overlapfams"],
    "clics4": ["family_count", "family_weight"]
}

for dataset, weightings in available_graphs.items():
    for weighting in weightings:
        if weighting == "affixfams":
            directed = to_undirected = True
        else:
            directed = to_undirected = False

        DATA_FP = GRAPH_DIR / dataset / f"{weighting}.json"
        OUTPUT_DIR = OUTPUT_BASE_DIR / dataset / weighting

        # SDNE
        sdne = SDNE.from_graph_file(DATA_FP, directed=directed, to_undirected=to_undirected)
        sdne.train(max_epochs=10000, patience=10, min_delta=0.001)
        sdne.save(OUTPUT_DIR / "sdne.json")
        print("Done training SDNE.")

        # Node2Vec (CBOW)
        node2vec = Node2Vec.from_graph_file(DATA_FP, directed=directed, to_undirected=to_undirected)
        node2vec.train(cbow=True, max_epochs=3000, patience=5, min_delta=0.001)
        node2vec.save(OUTPUT_DIR / "n2v-cbow.json")
        print("Done training Node2Vec (CBOW).")

        # Node2Vec (SkipGram)
        node2vec = Node2Vec.from_graph_file(DATA_FP, directed=directed, to_undirected=to_undirected)
        node2vec.train(cbow=False, max_epochs=1500, patience=5, min_delta=0.001)
        node2vec.save(OUTPUT_DIR / "n2v-sg.json")
        print("Done training Node2Vec (SkipGram).")

        # ProNE
        prone = ProNE.from_graph_file(DATA_FP, directed=directed, to_undirected=to_undirected)
        prone.train(embedding_size=128)
        prone.save(OUTPUT_DIR / "prone.json")
        print("Done training ProNE.")
