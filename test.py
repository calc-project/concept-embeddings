from graphembeddings.models.trainer import Node2Vec, ProNE, SDNE
from graphembeddings.utils.io import read_graph_data
from pathlib import Path


graph, id_to_concept, _ = read_graph_data(Path(__file__).parent / "data" / "graphs" / "clics4" / "family_count.json")

# test Node2Vec (CBOW)
node2vec = Node2Vec(graph, id_to_concept)
node2vec.train(cbow=True, max_epochs=100)
node2vec.save("n2v-cbow-test.json")
print("Done training Node2Vec (CBOW).")

# test Node2Vec (SkipGram)
node2vec = Node2Vec(graph, id_to_concept)
node2vec.train(cbow=False, max_epochs=100)
node2vec.save("n2v-sg-test.json")
print("Done training Node2Vec (SkipGram).")

# test ProNE
prone = ProNE(graph, id_to_concept)
prone.train()
prone.save("prone-test.json")
print("Done training ProNE.")

# test SDNE
sdne = SDNE(graph, id_to_concept)
sdne.train(max_epochs=100)
sdne.save("sdne-test.json")
print("Done training SDNE.")
