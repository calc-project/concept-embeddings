import numpy as np
import matplotlib.pyplot as plt
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


def extract_hyperparameters(names):
    hyperparameters = {}
    for name in names:
        param_dict = {}
        param_str = name.replace("semantic-node2vec-", "").replace("05", "0.5")
        fields = param_str.split("-")
        for i in range(0, len(fields), 2):
            parameter = fields[i]
            value = fields[i + 1]
            param_dict[parameter] = value
        hyperparameters[name] = param_dict

    return hyperparameters


def group_by_hyperparameter(metrics, param, hyperparameters):
    res = {}
    for name, value in metrics.items():
        try:
            hyperparam_value = hyperparameters[name][param]
            if hyperparam_value in res:
                res[hyperparam_value].append(value)
            else:
                res[hyperparam_value] = [value]
        except KeyError:
            raise ValueError(f"Name {name} or hyperparameter {param} does not exist.")

    sorted_res = sorted(res.items(), key=lambda x: float(x[0]))
    return [x[0] for x in sorted_res], [x[1] for x in sorted_res]

#test_emb = ConceptEmbeddings(encoder, "semantic-node2vec-p-2-q-2-n-100.npy")
#print(full_eval.eval_eat(test_emb.generate_embeddings(all_concepts)))


if __name__ == "__main__":
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
        msl_results[fn.stem] = full_eval.eval_msl(embeddings)
        #msl_results[fn] = babyclics_eval.eval_msl(embeddings)
        semshift_results[fn.stem] = full_eval.eval_semshift(embeddings)
        #semshift_results[fn] = babyclics_eval.eval_semshift(embeddings)
        eat_results[fn.stem] = full_eval.eval_eat(embeddings)
        #eat_results[fn] = babyclics_eval.eval_eat(embeddings)

    hyperparameters = extract_hyperparameters(msl_results.keys())
    for eval_name, eval_metric in {"msl": msl_results, "semshift": semshift_results, "eat": eat_results}.items():
        for hyperparam in ["p", "q", "n"]:
            param_values, metric_values = group_by_hyperparameter(eval_metric, hyperparam, hyperparameters)
            plt.cla()
            plt.violinplot(metric_values, data=param_values, showmeans=True)
            plt.title(f"{eval_name} by {hyperparam}")
            plt.xticks(range(len(param_values)+1), [""] + param_values)
            plt.show()

    #for metric in [msl_results, semshift_results, eat_results]:
    #    for k, v in sorted(metric.items(), key=lambda x: x[1], reverse=True):
    #        print(k, v)
    #    print(f"\n{100*'='}\n")
