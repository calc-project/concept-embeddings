import torch
import numpy as np
import networkx as nx
import json
from nodevectors import ProNE as ProNEEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from graphembeddings.models.nn import SDNEEmbedder, SDNELoss, CBOW, SkipGram

__all__ = ["SDNE", "Node2Vec", "ProNE"]


class GraphEmbeddingModel(object):
    DEFAULT_PARAMS = {
        "embedding_size": 128,
    }

    """
    Abstract class that defines the interface for graph embedding models.
    """
    def __init__(self, graph: np.ndarray, id_to_concept: dict):
        self.graph = graph  # as adjacency matrix
        self.id_to_concept = id_to_concept
        self.num_nodes = graph.shape[0]
        self.embeddings = None  # learned embeddings will be stored in this object
        self.callbacks = None  # can track certain metrics along training, if required
        self.training_params = {}

    def _get_training_params(self, **kwargs):
        """
        Compile the training parameters for this model by combining parameters given in **kwargs with default values.
        :param kwargs:
        :return:
        """
        return {param: kwargs.get(param, default) for param, default in self.DEFAULT_PARAMS.items()}

    def train(self, **kwargs):
        training_params = self._get_training_params(**kwargs)
        self.training_params = training_params
        self._train(**training_params)

    def _train(self, **kwargs):
        pass

    def save(self, fp):
        """
        :param fp: where to save the embeddings
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Train embeddings first.")

        data = {"parameters": self.training_params, "embeddings": self.embeddings}
        with open(fp, "w") as f:
            json.dump(data, f)


class ProNE(GraphEmbeddingModel):
    DEFAULT_PARAMS = {
        "embedding_size": 128,
    }

    def __init__(self, graph: np.ndarray, id_to_concept: dict):
        super().__init__(graph, id_to_concept)
        self.graph = nx.from_numpy_array(self.graph)

    def _train(self, **kwargs):
        embedding_size = kwargs.pop("embedding_size")
        model = ProNEEncoder(n_components=embedding_size, **kwargs)
        model.fit(self.graph)
        self.embeddings = {self.id_to_concept[id]: emb.tolist() for id, emb in model.model.items()}


class SDNE(GraphEmbeddingModel):
    DEFAULT_PARAMS = {
        "hidden_sizes": (256, 128),
        "alpha": 0.2,
        "beta": 10,
        "max_epochs": 1000,
        "lr": 1e-3,
        "weight_decay": 1e-5
    }

    def __init__(self, graph: np.ndarray, id_to_concept: dict):
        super().__init__(graph, id_to_concept)
        # generate L matrix
        self.D = np.diag(self.graph.sum(axis=1))
        self.L = self.D - self.graph

        # convert graph and L to torch tensors
        self.graph = torch.tensor(self.graph, dtype=torch.float32)
        self.L = torch.tensor(self.L, dtype=torch.float32)

    def _train(self, **kwargs):
        # get training parameters
        training_params = kwargs

        # set up model
        model = SDNEEmbedder(num_nodes=self.num_nodes, hidden_sizes=training_params["hidden_sizes"])

        # use custom loss function
        loss_function = SDNELoss(alpha=training_params["alpha"], beta=training_params["beta"])

        # compile optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"], weight_decay=training_params["weight_decay"])

        # store losses for early stopping
        losses = []
        best_loss = np.inf
        wait = 0
        patience = 0

        for epoch in tqdm(range(training_params["max_epochs"]), desc="Training SDNE..."):
            model.train()
            reconstructed, embedding = model(self.graph)

            # calculate loss
            loss = loss_function(self.graph, reconstructed, embedding, self.L)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss))

            # check for convergence
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if patience and wait > patience:
                    print(f"Training stopped after {epoch} epochs.")
                    break  # stop training

        # store embeddings
        self.embeddings = {self.id_to_concept[i]: model.embed(self.graph[i]).tolist() for i in range(self.num_nodes)}


class Node2Vec(GraphEmbeddingModel):
    """
    epochs = kwargs.get("epochs", 100)
    patience = kwargs.get("patience")
    delta = kwargs.get("delta", 0.0)
    test_split = kwargs.get("test_split", 0.2)
    shuffle = kwargs.get("shuffle", True)"""
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
        "lr": 1e-3
    }

    def random_walks_from_node(self, node, n=5, walk_length=10, p=1, q=1):
        walks = []

        for _ in range(n):
            walk = []
            alpha = None
            for _ in range(walk_length):
                # append current node to graph
                walk.append(node)

                # calculate probabilities for the next node
                neighbors = self.graph[node]
                if alpha:
                    scaled = alpha * neighbors
                    prob = scaled / scaled.sum()
                else:
                    prob = neighbors / neighbors.sum()

                # set up alpha (scaling vector) for the next iteration; based on the current node
                if not (p == 1 and q == 1):
                    alpha = np.where(neighbors != 0, 1, 1 / q)
                    alpha[node] = 1 / p

                # sample next node according to the probability distribution
                node = np.random.choice(self.num_nodes, p=prob)
            walks.append(walk)

        return walks

    def sample_random_walks(self, n=5, walk_length=10, p=1, q=1):
        walks = []

        for i in range(self.num_nodes):
            walks.extend(self.random_walks_from_node(i, n=n, walk_length=walk_length, p=p, q=q))

        return walks

    def generate_training_data(self, walks, window_size=2, cbow=True):
        """
        Generate training data from random walks.

        :param walks: the previously generated random walks
        :param window_size: the size of the context windows
        :param cbow: if True, generate training data for a CBOW model; otherwise for a SkipGram model.
        :return: the training data
        """
        target_nodes = []
        context_bow = []

        for walk in walks:
            for i, node in enumerate(walk):
                left_idx = max(0, i - window_size)
                right_idx = min(len(walk), i + window_size + 1)
                context = walk[left_idx:i] + walk[i+1:right_idx]
                target_nodes.append(node)
                context_bow.append(context)

        if cbow:
            for context in context_bow:
                while len(context) < window_size * 2:
                    context.append(self.num_nodes)  # padding token = next available token ID (i.e. the number of nodes)
            X = torch.tensor(context_bow)
            Y = torch.tensor(target_nodes)
        else:
            X, Y = [], []
            for node, context in zip(target_nodes, context_bow):
                for context_node in context:
                    X.append(node)
                    Y.append(context_node)
            X = torch.tensor(X)
            Y = torch.tensor(Y)

        return X, Y

    def _train(self, **kwargs):
        # read parameters
        training_params = kwargs
        cbow = training_params["cbow"]

        # generate training data from random walks
        walks = self.sample_random_walks(n=training_params["n"], walk_length=training_params["walk_length"],
                                         p=training_params["p"], q=training_params["q"])
        X, Y = self.generate_training_data(walks, window_size=training_params["window_size"], cbow=cbow)

        # split train and test data randomly
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=training_params["test_split"],
                                                            shuffle=training_params["shuffle"])

        # set up the model
        model = CBOW(self.num_nodes + 1, embed_dimension=training_params["embedding_size"]) if cbow \
            else SkipGram(self.num_nodes, embed_dimension=training_params["embedding_size"])

        # loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

        # callbacks for early stopping
        train_losses = []
        val_losses = []
        best_loss = np.inf
        wait = 0

        for epoch in tqdm(range(training_params["max_epochs"]), desc="Training Node2Vec..."):
            model.train()

            # forward pass
            pred = model(X_train)
            loss = criterion(pred, Y_train)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            with torch.no_grad():
                pred = model(X_test)
                val_loss = criterion(pred, Y_test)

            train_losses.append(float(loss))
            val_losses.append(float(val_loss))

            # check for convergence
            if val_loss.item() - best_loss < -training_params["min_delta"]:
                best_loss = val_loss.item()
                wait = 0
            else:
                wait += 1
                if training_params["patience"] is not None and wait > training_params["patience"]:
                    print(f"Training stopped after {epoch} epochs.")
                    break  # stop training

        embeddings = list(model.parameters())[0]
        self.embeddings = {self.id_to_concept[i]: embeddings[i].tolist() for i in self.id_to_concept}
