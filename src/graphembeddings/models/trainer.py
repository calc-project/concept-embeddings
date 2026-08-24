import torch
import numpy as np
import json
import datetime
import pickle
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split

from graphembeddings.models.nn import SDNEEmbedder, SDNELoss, CBOW, SkipGram, NCELoss
from graphembeddings.utils.io import read_graph_data
from graphembeddings.utils.preprocess import BOWEncoder, SBertEncoder

__all__ = ["SDNE", "Node2Vec", "ProNE", "SemanticNode2Vec", "SBertBaseline"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphEmbeddingModel(object):
    DEFAULT_PARAMS = {
        "embedding_size": 128,
    }

    def __init__(self, graph: np.ndarray, id_to_concept: dict, graph_data_fn: str, **kwargs):
        self.graph = graph
        self.id_to_concept = id_to_concept
        self.num_nodes = graph.shape[0]
        self.embeddings = None
        self.callbacks = None
        self.training_params = {"model": self.__class__.__name__, "training_data": graph_data_fn}
        self.device = DEVICE

    @classmethod
    def from_graph_file(cls, fp, directed=False, to_undirected=False):
        graph, id_to_concept, _, concept_coverage = read_graph_data(
            fp, directed=directed, to_undirected=to_undirected
        )
        kwargs = {"concept_coverage": concept_coverage}
        if isinstance(fp, Path):
            fp = "/".join(fp.parts[-2:])
        return cls(graph, id_to_concept, str(fp), **kwargs)

    def _get_training_params(self, **kwargs):
        return {param: kwargs.get(param, default) for param, default in self.DEFAULT_PARAMS.items()}

    def train(self, **kwargs):
        training_params = self._get_training_params(**kwargs)
        self.training_params.update(training_params)
        time_train_start = datetime.datetime.now()
        print(f"Training on device: {self.device}")
        self._train(**training_params)
        time_train_end = datetime.datetime.now()
        self.training_params["training_time"] = str(time_train_end - time_train_start)
        self.training_params["timestamp"] = time_train_end.strftime("%Y-%m-%d %H:%M:%S")

    def _train(self, **kwargs):
        pass

    def save(self, fp):
        if not self.embeddings:
            raise ValueError("No embeddings available. Train embeddings first.")
        data = {"parameters": self.training_params, "embeddings": self.embeddings}
        with open(fp, "w") as f:
            json.dump(data, f)


class ProNE(GraphEmbeddingModel):
    DEFAULT_PARAMS = {"embedding_size": 128}

    def __init__(self, graph: np.ndarray, id_to_concept: dict, graph_data_fp: str, **kwargs):
        super().__init__(graph, id_to_concept, graph_data_fp, **kwargs)

    def _train(self, **kwargs):
        pass


class SDNE(GraphEmbeddingModel):
    DEFAULT_PARAMS = {
        "hidden_sizes": (256, 128),
        "alpha": 0.2,
        "beta": 10,
        "max_epochs": 1000,
        "lr": 1e-3,
        "weight_decay": 1e-5,
    }

    def __init__(self, graph: np.ndarray, id_to_concept: dict, graph_data_fp: str, **kwargs):
        super().__init__(graph, id_to_concept, graph_data_fp, **kwargs)
        self.D = np.diag(self.graph.sum(axis=1))
        self.L = self.D - self.graph
        self.graph = torch.tensor(self.graph, dtype=torch.float32, device=self.device)
        self.L = torch.tensor(self.L, dtype=torch.float32, device=self.device)

    def _train(self, **kwargs):
        training_params = kwargs
        model = SDNEEmbedder(
            num_nodes=self.num_nodes, hidden_sizes=training_params["hidden_sizes"]
        ).to(self.device)
        loss_function = SDNELoss(alpha=training_params["alpha"], beta=training_params["beta"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_params["lr"]),
            weight_decay=training_params["weight_decay"],
        )
        best_loss = np.inf
        wait = 0
        patience = 0

        for epoch in tqdm(range(training_params["max_epochs"]), desc="Training SDNE..."):
            model.train()
            reconstructed, embedding = model(self.graph)
            loss = loss_function(self.graph, reconstructed, embedding, self.L)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if patience and wait > patience:
                    print(f"Training stopped after {epoch} epochs.")
                    break

        model.eval()
        with torch.no_grad():
            self.embeddings = {
                self.id_to_concept[i]: model.embed(self.graph[i]).detach().cpu().tolist()
                for i in range(self.num_nodes)
            }
        self.training_params["epochs"] = epoch + 1


class NodeContextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class Node2Vec(GraphEmbeddingModel):
    DEFAULT_PARAMS = {
        "embedding_size": 128,
        "cbow": True,
        "n": 5,
        "walk_length": 10,
        "p": 1,
        "q": 1,
        "window_size": 2,
        "max_epochs": 100,
        "patience": None,
        "min_delta": 0.00,
        "test_split": 0.2,
        "shuffle": True,
        "lr": 1e-3,
        "encodings": None,
        "ns": False,
        "ns_exponent": 1,
        "batch_size": 1028
    }

    def __init__(self, graph, id_to_concept: dict, graph_data_fp: str, **kwargs):
        """
        :param graph: a single np.ndarray or a list of np.ndarrays.
                      All matrices must have identical shape — use from_graph_files()
                      to load and align graphs with different concept sets automatically.
        :param kwargs:
            concept_coverages: list of per-graph frequency arrays (one per graph, each
                               re-indexed to the unified concept space).  Produced by
                               from_graph_files().  When supplied, each graph's own
                               coverage is used during its negative-sampling step.
            concept_coverage:  single coverage array (legacy / single-graph path).
                               Ignored when concept_coverages is present.
        """
        if isinstance(graph, np.ndarray):
            self.graphs = [graph]
        else:
            self.graphs = list(graph)

        shapes = {g.shape for g in self.graphs}
        if len(shapes) != 1:
            raise ValueError(
                f"All graphs must have the same shape, but got: {shapes}. "
                "Use from_graph_files() to align graphs with different concept sets."
            )

        super().__init__(self.graphs[0], id_to_concept, graph_data_fp, **kwargs)
        self.graph = None  # authoritative source is self.graphs

        # Per-graph coverages take priority; fall back to a single coverage wrapped in a list.
        if "concept_coverages" in kwargs and kwargs["concept_coverages"] is not None:
            self.concept_coverages = kwargs["concept_coverages"]
        elif "concept_coverage" in kwargs and kwargs["concept_coverage"] is not None:
            self.concept_coverages = [kwargs["concept_coverage"]]
        else:
            self.concept_coverages = None  # uniform sampling for NS

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_graph_files(cls, graph_configs):
        """
        Load and align multiple graph files into a single Node2Vec instance.

        :param graph_configs: iterable of dicts, one per graph, each supporting:
            - ``fp``             (required) – path to the graph file.
            - ``directed``       (optional, default ``False``)
            - ``to_undirected``  (optional, default ``False``)

        Example::

            Node2Vec.from_graph_files([
                {"fp": "graphs/colexification.csv"},
                {"fp": "graphs/similarity.csv",   "to_undirected": True},
                {"fp": "graphs/translation.csv",  "directed": True},
            ])

        Concept IDs are unified across all graphs.  Missing nodes are zero-padded.
        Coverage arrays are kept separate (one per graph) and re-indexed to the
        unified concept space.
        """
        raw_graphs, raw_id_to_concepts, raw_coverages = [], [], []
        fp_strs = []

        for cfg in graph_configs:
            fp = cfg["fp"]
            directed = cfg.get("directed", False)
            to_undirected = cfg.get("to_undirected", False)

            graph, id_to_concept, _, concept_coverage = read_graph_data(
                fp, directed=directed, to_undirected=to_undirected
            )
            raw_graphs.append(graph)
            raw_id_to_concepts.append(id_to_concept)
            raw_coverages.append(concept_coverage)
            fp_strs.append("/".join(Path(fp).parts[-2:]) if isinstance(fp, Path) else str(fp))

        aligned_graphs, unified_id_to_concept, per_graph_coverages = cls._align_graphs(
            raw_graphs, raw_id_to_concepts, raw_coverages
        )

        return cls(
            aligned_graphs,
            unified_id_to_concept,
            "+".join(fp_strs),
            concept_coverages=per_graph_coverages,
        )

    # ------------------------------------------------------------------
    # Graph alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_graphs(graphs, id_to_concepts, concept_coverages):
        """
        Re-index a collection of graphs to a shared concept space.

        Coverage arrays are re-indexed individually and returned as a list —
        one array per source graph — rather than being summed.  Concepts absent
        from a particular graph receive a coverage value of 0 in that graph's array.

        :param graphs: list of np.ndarray adjacency matrices (possibly different sizes).
        :param id_to_concepts: list of {int -> str} dicts, one per graph.
        :param concept_coverages: list of per-node frequency arrays (or None), one per graph.
        :returns: (aligned_graphs, unified_id_to_concept, per_graph_coverages)
                  where per_graph_coverages is a list[np.ndarray | None].
        """
        all_concepts = sorted({c for itc in id_to_concepts for c in itc.values()})
        unified_concept_to_id = {c: i for i, c in enumerate(all_concepts)}
        unified_id_to_concept = dict(enumerate(all_concepts))
        num_nodes = len(all_concepts)

        aligned_graphs = []
        per_graph_coverages = []

        for graph, id_to_concept, coverage in zip(graphs, id_to_concepts, concept_coverages):
            old_to_new = np.array(
                [unified_concept_to_id[id_to_concept[old_i]] for old_i in range(len(id_to_concept))],
                dtype=int,
            )

            # Re-index adjacency matrix
            new_graph = np.zeros((num_nodes, num_nodes), dtype=graph.dtype)
            rows, cols = np.nonzero(graph)
            new_graph[old_to_new[rows], old_to_new[cols]] = graph[rows, cols]
            aligned_graphs.append(new_graph)

            # Re-index this graph's coverage into the unified space (zeros for absent nodes)
            if coverage is not None:
                new_coverage = np.zeros(num_nodes, dtype=float)
                new_coverage[old_to_new] = np.array(coverage, dtype=float)
                per_graph_coverages.append(new_coverage)
            else:
                per_graph_coverages.append(None)

        return aligned_graphs, unified_id_to_concept, per_graph_coverages

    # ------------------------------------------------------------------
    # Random walk sampling
    # ------------------------------------------------------------------

    def random_walks_from_node(self, node, graph: np.ndarray, n=5, walk_length=10, p=1, q=1):
        walks = []
        for _ in range(n):
            walk = []
            alpha = None
            current = node
            for _ in range(walk_length):
                walk.append(current)
                neighbors = graph[current]
                neighbor_sum = neighbors.sum()
                if neighbor_sum == 0:
                    break
                prob = (
                    (alpha * neighbors) / (alpha * neighbors).sum()
                    if alpha is not None
                    else neighbors / neighbor_sum
                )
                if not (p == 1 and q == 1):
                    alpha = np.where(neighbors != 0, 1.0, 1.0 / q)
                    alpha[current] = 1.0 / p
                current = np.random.choice(self.num_nodes, p=prob)
            walks.append(walk)
        return walks

    def sample_random_walks(self, n=5, walk_length=10, p=1, q=1):
        """
        Sample random walks across all stored graphs.

        :param n: walks per node.  An ``int`` broadcasts to every graph; a tuple/list
                  supplies one count per graph and must match ``len(self.graphs)``.
        """
        if isinstance(n, int):
            n_per_graph = [n] * len(self.graphs)
        else:
            n_per_graph = list(n)
            if len(n_per_graph) != len(self.graphs):
                raise ValueError(
                    f"n has {len(n_per_graph)} entries but there are {len(self.graphs)} graphs. "
                    "Provide one value per graph or a single int to broadcast."
                )

        walks = []
        for graph, n_g in zip(self.graphs, n_per_graph):
            if n_g == 0:
                continue
            for i in tqdm(range(self.num_nodes), desc=f"Sampling random walks..."):
                if graph[i].sum() == 0:
                    continue
                walks.extend(
                    self.random_walks_from_node(i, graph, n=n_g, walk_length=walk_length, p=p, q=q)
                )
        return walks

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------

    def generate_training_data(self, walks, window_size=2, cbow=True, encodings=None,
                               ns=False, ns_exponent=1):
        """
        :param walks: random walks (mixed across all graphs).
        :param ns: if True, build noise-contrastive targets.  When concept_coverages is
                   available the per-graph arrays are summed to form a single sampling
                   distribution, since NS operates over the pooled walk vocabulary rather
                   than any individual graph.
        """
        print(f"{ns_exponent=}")

        target_nodes, context_bow = [], []

        for walk in walks:
            for i, node in enumerate(walk):
                left_idx = max(0, i - window_size)
                right_idx = min(len(walk), i + window_size + 1)
                context = walk[left_idx:i] + walk[i + 1:right_idx]
                target_nodes.append(node)
                context_bow.append(context)

        if cbow:
            for context in context_bow:
                while len(context) < window_size * 2:
                    context.append(self.num_nodes)
            #X = torch.tensor(context_bow, device=self.device)
            #Y = torch.tensor(target_nodes, device=self.device)
        else:
            X, Y = [], []
            for node, context in zip(target_nodes, context_bow):
                for context_node in context:
                    X.append(encodings[node] if encodings else node)
                    Y.append(context_node)
            #X = torch.tensor(X, device=self.device)
            #if encodings:
            #    X = X.to(torch.float32)
            #Y = torch.tensor(Y, device=self.device)

        if ns:
            # Derive a single pooled distribution from per-graph coverages by summing them.
            # Each graph's array has already been zero-padded to the unified concept space,
            # so summing is safe and preserves relative frequencies across graphs.
            if self.concept_coverages is not None:
                valid = [c for c in self.concept_coverages if c is not None]
                pooled_coverage = np.sum(valid, axis=0) if valid else None
            else:
                pooled_coverage = None

            y_nodes = Y.copy()
            Y = []
            for node in y_nodes:
                y_vec = [0] * self.num_nodes
                y_vec[node] = 1
                counts = (
                    pooled_coverage ** ns_exponent
                    if pooled_coverage is not None
                    else np.ones(self.num_nodes)
                )
                counts[node] = 0
                for _ in range(5):
                    distractor = random.sample(
                        range(self.num_nodes), 1, counts=counts.astype("int").tolist()
                    )[0]
                    y_vec[distractor] = -1
                    counts[distractor] = 0
                Y.append(y_vec)
            # Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        return NodeContextDataset(X, Y)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train(self, **kwargs):
        training_params = kwargs
        cbow = training_params["cbow"]
        encodings = training_params["encodings"]
        bow_input_dim = len(encodings[0]) if encodings else None
        ns = training_params["ns"]

        walks = self.sample_random_walks(
            n=training_params["n"],
            walk_length=training_params["walk_length"],
            p=training_params["p"],
            q=training_params["q"],
        )
        dataset = self.generate_training_data(
            walks,
            window_size=training_params["window_size"],
            cbow=cbow,
            encodings=encodings,
            ns=ns,
            ns_exponent=training_params["ns_exponent"],
        )
        test_split=training_params["test_split"]
        train_set, test_set = random_split(dataset, [1-test_split, test_split])
        print(f"batch_size: {training_params.get("batch_size", 1028)}")
        train_dataloader = DataLoader(train_set, batch_size=training_params.get("batch_size", 1028), shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=training_params.get("batch_size", 1028), shuffle=True)

        model = (
            CBOW(self.num_nodes + 1, embed_dimension=training_params["embedding_size"])
            if cbow
            else SkipGram(
                self.num_nodes,
                embed_dimension=training_params["embedding_size"],
                bow_input_dim=bow_input_dim,
            )
        )
        model = model.to(self.device)

        criterion = NCELoss() if ns else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training_params["lr"]))

        best_loss = np.inf
        wait = 0
        submodel_str = "CBOW" if cbow else "SkipGram"
        training_progress = tqdm(
            range(training_params["max_epochs"]),
            desc=f"Training Node2Vec... ({submodel_str})",
        )
        for epoch in training_progress:
            model.train()
            epoch_loss = 0
            for X_train, Y_train in train_dataloader:
                X_train = X_train.to(self.device)
                Y_train = Y_train.to(self.device)
                pred = model(X_train)
                loss = criterion(pred, Y_train)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for X_test, Y_test in test_dataloader:
                X_test = X_test.to(self.device)
                Y_test = Y_test.to(self.device)
                with torch.no_grad():
                    val_loss = criterion(model(X_test), Y_test)

            training_progress.set_description(
                f"Training Node2Vec... ({submodel_str}) | "
                f"Loss (train): {float(loss):.4f} | Loss (val): {float(val_loss):.4f}"
            )

            if val_loss.item() - best_loss < -training_params["min_delta"]:
                best_loss = val_loss.item()
                wait = 0
            else:
                wait += 1
                if training_params["patience"] is not None and wait > training_params["patience"]:
                    print(f"Training stopped after {epoch} epochs.")
                    break

        model.eval()
        if encodings:
            with torch.no_grad():
                self.embeddings = {
                    c: model.embeddings(
                        torch.tensor(encodings[i], device=self.device).to(torch.float32)
                    ).detach().cpu().tolist()
                    for i, c in self.id_to_concept.items()
                }
            self.embedding_weights = model.embeddings
        else:
            embeddings = list(model.parameters())[0]
            self.embeddings = {
                self.id_to_concept[i]: embeddings[i].detach().cpu().tolist()
                for i in self.id_to_concept
            }

        self.training_params["epochs"] = epoch + 1
        self.training_params["num_graphs"] = len(self.graphs)


class SemanticNode2Vec(GraphEmbeddingModel):
    DEFAULT_PARAMS = {
        "embedding_size": 128,
        "cbow": True,
        "n": 5,
        "walk_length": 10,
        "p": 1,
        "q": 1,
        "window_size": 2,
        "max_epochs": 100,
        "patience": None,
        "min_delta": 0.00,
        "test_split": 0.2,
        "shuffle": True,
        "lr": 1e-3,
        "keep_one_hot": False,
        "min_token_count": 2,
        "encoder": "bow",
        "ns": False,
        "ns_exponent": 1,
        "batch_size": 1028
    }

    def __init__(self, graph, id_to_concept: dict, graph_data_fn: str, **kwargs):
        if isinstance(graph, np.ndarray):
            self.graphs = [graph]
        else:
            self.graphs = list(graph)

        super().__init__(self.graphs[0], id_to_concept, graph_data_fn, **kwargs)
        self.graph = None

        if "concept_coverages" in kwargs and kwargs["concept_coverages"] is not None:
            self.concept_coverages = kwargs["concept_coverages"]
        elif "concept_coverage" in kwargs and kwargs["concept_coverage"] is not None:
            self.concept_coverages = [kwargs["concept_coverage"]]
        else:
            self.concept_coverages = None

    @classmethod
    def from_graph_files(cls, graph_configs):
        """
        Load and align multiple graph files.  Accepts the same ``graph_configs`` format
        as ``Node2Vec.from_graph_files`` (list of dicts with ``fp``, ``directed``,
        ``to_undirected``).  Delegates alignment to ``Node2Vec._align_graphs``.
        """
        raw_graphs, raw_id_to_concepts, raw_coverages = [], [], []
        fp_strs = []

        for cfg in graph_configs:
            fp = cfg["fp"]
            directed = cfg.get("directed", False)
            to_undirected = cfg.get("to_undirected", False)

            graph, id_to_concept, _, concept_coverage = read_graph_data(
                fp, directed=directed, to_undirected=to_undirected
            )
            raw_graphs.append(graph)
            raw_id_to_concepts.append(id_to_concept)
            raw_coverages.append(concept_coverage)
            fp_strs.append("/".join(Path(fp).parts[-2:]) if isinstance(fp, Path) else str(fp))

        aligned_graphs, unified_id_to_concept, per_graph_coverages = Node2Vec._align_graphs(
            raw_graphs, raw_id_to_concepts, raw_coverages
        )

        return cls(
            aligned_graphs,
            unified_id_to_concept,
            "-".join(fp_strs),
            concept_coverages=per_graph_coverages,
        )

    def _train(self, **kwargs):
        min_token_count = self.training_params.pop("min_token_count")
        keep_one_hot = self.training_params.pop("keep_one_hot")
        print(kwargs["encoder"])

        self.node2vec = Node2Vec(
            self.graphs,
            self.id_to_concept,
            self.training_params["training_data"],
            concept_coverages=self.concept_coverages,
        )

        if kwargs["encoder"] == "bow":
            self.encoder = BOWEncoder(
                list(self.id_to_concept.values()),
                min_token_count=min_token_count,
                keep_one_hot=keep_one_hot,
            )
        else:
            self.encoder = SBertEncoder(list(self.id_to_concept.values()))

        concept_to_id = {c: i for i, c in self.id_to_concept.items()}
        encodings = self.encoder.generate_encoding_matrix(concept_to_id)

        self.node2vec.train(**kwargs, encodings=encodings)
        self.embeddings = self.node2vec.embeddings
        if "encodings" in self.node2vec.training_params:
            self.node2vec.training_params.pop("encodings")
        self.training_params = self.node2vec.training_params

    def save(self, fp):
        super().save(fp)
        if isinstance(fp, str):
            fp = Path(fp)
        pkl_path = Path(fp.parent / fp.stem).with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    def embed(self, concept):
        if concept in self.embeddings:
            return self.embeddings[concept]
        input_vec = self.encoder.encode_concept(concept)
        input_vec = torch.tensor(input_vec, device=self.node2vec.device).to(torch.float32)
        with torch.no_grad():
            return self.node2vec.embedding_weights(input_vec).detach().cpu().tolist()

    def inductive_embeddings(self):
        missing_concepts = [
            x.gloss for x in self.encoder.con.conceptsets.values()
            if x.gloss not in self.encoder.concepts
        ]
        return {concept: self.embed(concept) for concept in missing_concepts}


class SBertBaseline(GraphEmbeddingModel):
    def _train(self, **kwargs):
        concepts = list(self.id_to_concept.values())
        self.encoder = SBertEncoder(concepts)
        self.embeddings = {c: self.encoder.encode_concept(c) for c in concepts}

    def inductive_embeddings(self):
        missing_concepts = [
            x.gloss for x in self.encoder.con.conceptsets.values()
            if x.gloss not in self.encoder.concepts
        ]
        return {c: self.encoder.encode_concept(c) for c in missing_concepts}

    def save(self, fp):
        super().save(fp)
        if isinstance(fp, str):
            fp = Path(fp)
        pkl_path = Path(fp.parent / fp.stem).with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)
