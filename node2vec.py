import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


class CBOW(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781

    Taken from https://github.com/OlgaChernytska/word2vec-pytorch/blob/main/utils/model.py (with slight modifications)
    """
    def __init__(self, vocab_size: int, embed_dimension: int = 128, padding=True):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=1,
            padding_idx=(vocab_size - 1) if padding else None
        )
        self.linear = nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781

    Taken from https://github.com/OlgaChernytska/word2vec-pytorch/blob/main/utils/model.py (with slight modifications)
    """
    def __init__(self, vocab_size: int, embed_dimension: int = 128):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=1,
        )
        self.linear = nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x


class Node2Vec(object):
    def __init__(self, graph: np.ndarray):
        self.graph = graph  # as adjacency matrix (i guess?)
        self.num_nodes = graph.shape[0]
        self.embeddings = None

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

    def train(self, n=5, walk_length=10, p=1, q=1, window_size=2, cbow=True, **kwargs):
        # read parameters (TODO refine)
        epochs = kwargs.get("epochs", 100)
        patience = kwargs.get("patience")
        delta = kwargs.get("delta", 0.0)
        test_split = kwargs.get("test_split", 0.2)
        shuffle = kwargs.get("shuffle", True)

        # generate training data from random walks
        walks = self.sample_random_walks(n=n, walk_length=walk_length, p=p, q=q)
        X, Y = self.generate_training_data(walks, window_size=window_size, cbow=cbow)

        # split train and test data randomly
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split, shuffle=shuffle)

        # set up the model
        model = CBOW(self.num_nodes + 1) if cbow else SkipGram(self.num_nodes)

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # callbacks for early stopping
        train_losses = []
        val_losses = []
        best_loss = np.inf
        wait = 0

        # TODO temporary check, remove later
        # load multisimlex rating for evaluation
        msl = msl_ratings()
        msl_losses = []

        for pair in set(msl.keys()):
            c1, c2 = pair
            if not (c1 in concept_to_id and c2 in concept_to_id):
                msl.pop(pair)

        # TODO smarter training (properly pass down parameters, batch training, early stopping, etc)
        for epoch in range(epochs):
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

            # MSL validation
            msl_similarities = []
            pred_similarities = []
            embeddings = list(model.parameters())[0]
            for pair, sim in msl.items():
                c1, c2 = pair
                idx1, idx2 = concept_to_id[c1], concept_to_id[c2]
                emb1 = embeddings[idx1].detach().numpy()
                emb2 = embeddings[idx2].detach().numpy()
                # cosine similarity between embeddings
                pred_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                msl_similarities.append(sim)
                pred_similarities.append(pred_sim)

            msl_corr = float(spearmanr(msl_similarities, pred_similarities).statistic)
            msl_losses.append(msl_corr)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Training loss: {loss.item():.4f}, Validation loss: {val_loss.item():.4f}, "
                      f"MSL: {msl_corr:.4f}")

            # check for convergence
            if val_loss.item() - best_loss < -delta:
                best_loss = val_loss.item()
                wait = 0
            else:
                wait += 1
                if patience is not None and wait > patience:
                    print(f"Training stopped after {epoch} epochs.")
                    break  # stop training

        self.embeddings = list(model.parameters())[0]
        # plot_losses(val_losses)
        plt.cla()
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Losses')

        # dirty fix for better visualization
        msl_losses = np.array(msl_losses) * 10

        plt.plot(train_losses, color='b', label='Training loss')
        plt.plot(val_losses, color='r', label='Validation loss')
        plt.plot(msl_losses, color='g', label='MSL correlation')
        plt.show()


def plot_losses(losses):
    # defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.plot(losses)
    plt.show()


# TODO obviously this doesn't belong here
def read_network_file(edgelist_file="clics-edgelist.tsv"):
    with open(edgelist_file) as f:
        rows = f.read().split("\n")

    assert rows[0].startswith("# ")

    num_nodes = int(rows[0].replace("# ", ""))
    graph = np.zeros((num_nodes, num_nodes))

    for row in rows[1:]:
        if not row:
            continue
        i, j, weight = row.split("\t")
        i = int(i)
        j = int(j)
        weight = float(weight)

        graph[i, j] = weight
        graph[j, i] = weight

    return graph

def msl_ratings(fp="multisimlex.csv"):
    # load multisimlex ratings
    msl = {}

    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept1 = row["CGL1"]
            concept2 = row["CGL2"]
            mean_similarity = float(row["mean"])
            if concept1 != concept2:
                msl[(concept1, concept2)] = mean_similarity

    return msl

# load idx to concept
id_dict = {}
# ...and concept to idx
concept_to_id = {}

with open("clics4/concept-ids-Family_Count.tsv") as f:
    for row in f.read().split("\n"):
        if not row:
            continue
        idx, concept = row.split("\t")
        id_dict[int(idx)] = concept
        concept_to_id[concept] = int(idx)


if __name__ == "__main__":
    graph = read_network_file("clics4/edgelist-Family_Count.tsv")
    node2vec = Node2Vec(graph)
    node2vec.train(epochs=10000, patience=5, delta=0.001)
    # node2vec.train(cbow=False)

    with open("embeddings/2025-01-07/n2v-cbow-w.tsv", "w") as f:
        for i, emb in enumerate(node2vec.embeddings[:-1]):
            concept = id_dict[i]
            f.write(f"{concept}\t{[float(x) for x in list(emb)]}\n")

    node2vec = Node2Vec(graph)
    node2vec.train(epochs=10000, patience=5, delta=0.001, cbow=False)

    with open("embeddings/2025-01-07/n2v-sg-w.tsv", "w") as f:
        for i, emb in enumerate(node2vec.embeddings):
            concept = id_dict[i]
            f.write(f"{concept}\t{[float(x) for x in list(emb)]}\n")
