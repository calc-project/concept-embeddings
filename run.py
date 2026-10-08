import argparse
import numpy as np
import yaml
from itertools import product
from pathlib import Path

from graphembeddings.models.trainer import Node2Vec, SemanticNode2Vec, SBertBaseline


MODEL_REGISTRY = {
    "node2vec": Node2Vec,
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

                out_dir = Path(output_base) / graph_name
                out_dir.mkdir(parents=True, exist_ok=True)

                # Determine model class
                model_key = model_cfg.get("class", model_name)
                ModelClass = MODEL_REGISTRY[model_key]
                out_fp = Path(out_dir) / f"{model_name}.json"

                if out_fp.exists() and not config.get("retrain", False):
                    print(f"Loaded {model_name} on {graph_name}.")
                else:
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
                            continue

                    train_kwargs = model_cfg.get("train", {})
                    train_kwargs.update(hyperparams)
                    if train_kwargs.get("ns_exponent", 0) < 0:
                        train_kwargs["ns"] = False
                    if not multi_graph and isinstance(train_kwargs.get("n"), list):
                        train_kwargs["n"] = sum(train_kwargs["n"])
                    model.train(**train_kwargs)
                    np.savetxt(out_dir / f"{local_model_name}.txt", model.node2vec.embedding_weights[0].weight.detach().cpu().numpy(), allow_pickle=False)
                    print(f"Saved {local_model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)