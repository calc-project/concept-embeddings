from pathlib import Path
from pyconcepticon import Concepticon
from tabulate import tabulate

from conceptembeddings.pretrained import ConceptEmbeddings
from conceptembeddings.preprocess import SBertEncoder
from graphembeddings.eval.eval import Evaluation
from graphembeddings.utils.io import read_graph_data, read_embeddings


BASE_DIR = Path(__file__).parent.parent

# set up concepticon and inductive models
con = Concepticon(BASE_DIR / "concepticon-data")
all_concepts = [x.gloss for x in con.conceptsets.values()]
model = ConceptEmbeddings(concepticon=con)
sbert_baseline = SBertEncoder(all_concepts, con=con)

# load IDS concepts
_, _, ids_concepts, _ = read_graph_data(BASE_DIR / "data/graphs/fullfams.json")
ids_concepts = list(ids_concepts.keys())
_, _, ids_affix_concepts, _ = read_graph_data(BASE_DIR / "data/graphs/overlapfams.json")
ids_affix_concepts = list(ids_affix_concepts.keys())
ids_concepts = [x for x in ids_concepts if x in ids_affix_concepts]
print(len(ids_concepts))

# load concepts from CLICS4 and CLIPS
_, _, clics_concepts, _ = read_graph_data(BASE_DIR / "data/graphs/clics4.json")
_, _, clips_concepts, _ = read_graph_data(BASE_DIR / "data/graphs/clips.json")
clics_clips_concepts = list(clics_concepts.keys() | clips_concepts.keys())
unseen_concepts = [c for c in all_concepts if c not in clics_clips_concepts]

###############################################################
################### TRANSDUCTIVE EVALUATION ###################
###############################################################

transductive_eval = Evaluation(ids_concepts)
prone_embeddings = read_embeddings(BASE_DIR / "transductive-embeddings/full-affix/prone.json")
node2vec_embeddings = read_embeddings(BASE_DIR / "transductive-embeddings/full-affix/n2v-sg.json")
semantic_node2vec_embeddings = model.generate_embeddings(ids_concepts)
baseline_embeddings = {c: sbert_baseline.encode_concept(c) for c in ids_concepts}
table = [list(transductive_eval.eval_all(emb)) for emb in
         (prone_embeddings, node2vec_embeddings, semantic_node2vec_embeddings, baseline_embeddings)]
print(tabulate(table, headers=["LSIM", "Semantic Change", "Word Associations"], showindex=["ProNE", "Node2Vec", "Semantic Node2Vec", "S-BERT Baseline"], tablefmt="latex", floatfmt=".2f"))


###############################################################
#################### INDUCTIVE EVALUATION #####################
###############################################################
table = []
for conceptset in [clics_clips_concepts, unseen_concepts, all_concepts]:
    eval = Evaluation(conceptset)
    embeddings = model.generate_embeddings(conceptset)
    table.append(list(eval.eval_all(embeddings)))
print(tabulate(table, ["LSIM", "Semantic Change", "Word Associations"], showindex=["seen concepts", "unseen concepts", "all concepts"], tablefmt="latex", floatfmt=".2f"))
