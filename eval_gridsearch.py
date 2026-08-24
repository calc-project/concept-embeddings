import numpy as np
from graphembeddings.eval.eval import Evaluation
from graphembeddings.utils.preprocess import SBertEncoder
from graphembeddings.utils.io import read_graph_data
from pyconcepticon import Concepticon
from pathlib import Path


SAVED_WEIGHTS_DIR = Path(__file__).parent / "output" / "gridsearch" / "clics4-clips"

class ConceptEmbeddings(object):
    def __init__(self, encoder, weights_fn):
        self.encoder = encoder
        self.weights = self._load_weights(weights_fn)

    def _load_weights(self, weights_fn):
        with open(SAVED_WEIGHTS_DIR / weights_fn, "rb") as f:
            weights = np.load(f).transpose()
        return weights

    def __call__(self, concept):
        return np.matmul(self.encoder.encode_concept(concept), self.weights)

    def generate_embeddings(self, concepts):
        return {c: self(c) for c in concepts}


#test_emb = ConceptEmbeddings(encoder, "semantic-node2vec-p-2-q-2-n-100.npy")
#print(full_eval.eval_eat(test_emb.generate_embeddings(all_concepts)))

con = Concepticon()
all_concepts = [x.gloss for x in con.conceptsets.values()]
full_eval = Evaluation(all_concepts)
encoder = SBertEncoder(all_concepts, con=con)
_, _, babyclics_concepts, _ = read_graph_data("data/graphs/fullfams.json")
babyclics_concepts = list(babyclics_concepts.keys())
_, _, babyclics_affix_concepts, _ = read_graph_data("data/graphs/overlapfams.json")
babyclics_affix_concepts = list(babyclics_affix_concepts.keys())
babyclics_concepts = [x for x in babyclics_concepts if x in babyclics_affix_concepts]
babyclics_eval = Evaluation(babyclics_concepts)

msl_results = {}
semshift_results = {}
eat_results = {}

for fn in SAVED_WEIGHTS_DIR.glob("*.npy"):
    emb = ConceptEmbeddings(encoder, fn)
    embeddings = emb.generate_embeddings(all_concepts)
    #msl_results[fn] = full_eval.eval_msl(embeddings)
    msl_results[fn] = babyclics_eval.eval_msl(embeddings)
    #semshift_results[fn] = full_eval.eval_semshift(embeddings)
    semshift_results[fn] = babyclics_eval.eval_semshift(embeddings)
    #eat_results[fn] = full_eval.eval_eat(embeddings)
    eat_results[fn] = babyclics_eval.eval_eat(embeddings)

for metric in [msl_results, semshift_results, eat_results]:
    for k, v in sorted(metric.items(), key=lambda x: x[1], reverse=True):
        print(k, v)
    print(f"\n{100*'='}\n")
