import argparse
import pickle
import numpy as np
from collections import defaultdict
from itertools import product

import yaml
from pathlib import Path
from tabulate import tabulate
from pyconcepticon import Concepticon

from graphembeddings.models.trainer import Node2Vec, ProNE, SDNE, SemanticNode2Vec, SBertBaseline
from graphembeddings.utils.io import read_graph_data, read_embeddings
from graphembeddings.eval.eval import Evaluation


MODEL_REGISTRY = {
    "sdne": SDNE,
    "node2vec": Node2Vec,
    "prone": ProNE,
    "semantic-node2vec": SemanticNode2Vec,
    "sbert-baseline": SBertBaseline
}

SEMANTIC_MODELS = ["semantic-node2vec", "sbert-baseline"]


def generate_hyperparameter_grid(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def update_model_name(base_name, hyperparams):
    name = base_name
    for key, value in hyperparams.items():
        if isinstance(value, list):
            value = value[0]
        value = str(value).replace(" ", "").replace(".", "")
        name += f"-{key}-{value}"

    return name


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    input_base = Path(config["input_base_dir"])
    output_base = Path(config["output_base_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    eval = config["eval"]
    if eval:
        eval_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        con = Concepticon()

    for graph_name, graph_cfg in config["graphs"].items():
        multi_graph = False
        graph_fp = graph_configs = None
        # case 1 - single graph
        if "directed" in graph_cfg:
            graph_fp = Path(input_base) / f"{graph_name}.json"
        else: # case 2 - multiple graphs
            graph_configs = []
            for name, cfg in graph_cfg.items():
                graph_fp = Path(input_base) / f"{name}.json"
                cfg["fp"] = graph_fp
                graph_configs.append(cfg)
            multi_graph = True

        for model_name, model_cfg in config["models"].items():
            hyperparam_definitions = config.get("hyperparameters", {})
            for hyperparams in generate_hyperparameter_grid(**hyperparam_definitions):
                local_model_name = model_name
                train = False

                out_dir = Path(output_base) / graph_name
                out_dir.mkdir(parents=True, exist_ok=True)

                # Determine model class
                model_key = model_cfg.get("class", model_name)
                ModelClass = MODEL_REGISTRY[model_key]
                out_fp = Path(out_dir) / f"{model_name}.json"

                if out_fp.exists() and not config.get("retrain", False):
                    embeddings = read_embeddings(out_fp)
                    print(f"Loaded {model_name} on {graph_name}.")
                else:
                    train = True
                    if hyperparams:
                        local_model_name = update_model_name(model_name, hyperparams)
                        print(hyperparams)
                    print(f"Training {local_model_name} on {graph_name} ...")

                    if not multi_graph:
                        model = ModelClass.from_graph_file(
                            graph_fp,
                            directed=graph_cfg.get("directed", False),
                            to_undirected=graph_cfg.get("to_undirected", False),
                        )
                    else:
                        if "node2vec" in model_key:
                            model = ModelClass.from_graph_files(graph_configs)
                        else:
                            if eval:
                                eval_metrics["msl"][model_name][graph_name] = 0
                                eval_metrics["semshift"][model_name][graph_name] = 0
                                eval_metrics["eat"][model_name][graph_name] = 0
                                if model_key in SEMANTIC_MODELS:
                                    # inductive
                                    eval_metrics["msl-inductive"][model_name][graph_name] = 0
                                    eval_metrics["semshift-inductive"][model_name][graph_name] = 0
                                    eval_metrics["eat-inductive"][model_name][graph_name] = 0
                                    # combined
                                    eval_metrics["msl-combined"][model_name][graph_name] = 0
                                    eval_metrics["semshift-combined"][model_name][graph_name] = 0
                                    eval_metrics["eat-combined"][model_name][graph_name] = 0
                            continue

                    train_kwargs = model_cfg.get("train", {})
                    train_kwargs.update(hyperparams)
                    if train_kwargs.get("ns_exponent", 0) < 0:
                        train_kwargs["ns"] = False
                    if not multi_graph and isinstance(train_kwargs.get("n"), list):
                        train_kwargs["n"] = sum(train_kwargs["n"])
                    model.train(**train_kwargs)
                    np.save(out_dir / f"{local_model_name}.npy", model.node2vec.embedding_weights[0].weight.detach().cpu().numpy(), allow_pickle=False)
                    # model.save(out_dir / f"{model_name}.json")
                    print(f"Saved {local_model_name}")

                    embeddings = model.embeddings

                concepts = list(embeddings.keys())

                if eval:
                    evaluation = Evaluation(concepts)
                    if model_key in SEMANTIC_MODELS:
                        if not train:
                            with open(out_dir / f"{model_name}.pkl", "rb") as f:
                                model = pickle.load(f)

                        missing_concepts = [x.gloss for x in con.conceptsets.values() if x.gloss not in concepts]
                        inductive_evaluation = Evaluation(missing_concepts)
                        combined_evaluation = Evaluation(concepts + missing_concepts)

                        inductive_embeddings = model.inductive_embeddings()
                        combined_embeddings = model.embeddings | inductive_embeddings
                        # transductive
                        msl, semshift, eat = evaluation.eval_all(model.embeddings)
                        eval_metrics["msl"][model_name][graph_name] = msl
                        eval_metrics["semshift"][model_name][graph_name] = semshift
                        eval_metrics["eat"][model_name][graph_name] = eat
                        # inductive
                        msl, semshift, eat = inductive_evaluation.eval_all(inductive_embeddings)
                        eval_metrics["msl-inductive"][model_name][graph_name] = msl
                        eval_metrics["semshift-inductive"][model_name][graph_name] = semshift
                        eval_metrics["eat-inductive"][model_name][graph_name] = eat
                        # combined
                        msl, semshift, eat = combined_evaluation.eval_all(combined_embeddings)
                        eval_metrics["msl-combined"][model_name][graph_name] = msl
                        eval_metrics["semshift-combined"][model_name][graph_name] = semshift
                        eval_metrics["eat-combined"][model_name][graph_name] = eat
                    else:
                        # only transductive evaluation
                        msl, semshift, eat = evaluation.eval_all(embeddings)
                        eval_metrics["msl"][model_name][graph_name] = msl
                        eval_metrics["semshift"][model_name][graph_name] = semshift
                        eval_metrics["eat"][model_name][graph_name] = eat

    if eval:
        for eval_method, metrics in eval_metrics.items():
            models = list(metrics.keys())
            graphs = list(metrics[models[0]].keys())
            table = []
            for model in models:
                row = [metrics[model][graph] for graph in graphs]
                table.append(row)
            print(tabulate(table, headers=[eval_method] + graphs, showindex=models, tablefmt="fancy_grid", floatfmt=".4f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)