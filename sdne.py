import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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


concept_to_id = {}
id_to_concept = {}

with open("clics4/concept-ids-Family_Count.tsv") as f:
    for row in f:
        if not row:
            continue
        id, concept = row.strip().split("\t")
        id = int(id)
        concept_to_id[concept] = id
        id_to_concept[id] = concept


class SDNELoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(SDNELoss, self).__init__()
        self.alpha = kwargs.get('alpha', 1e-6)
        self.beta = kwargs.get('beta', 5.)

    def forward(self, X, X_hat, Y, L):
        """
        Loss function as described in Wang et al. (2016).

        Note: L2 regularization is omitted here, since it is passed directly to the optimizer (as weight_decay)
        :param X: the actual adjacency matrix
        :param X_hat: the reconstructed adjacency matrix
        :param Y: the embedding matrix
        :param L: the L matrix (Eq. 9)
        :return: the loss
        """
        # calculate L_1st (see Eq. 9)
        l_first = 2 * (Y.transpose(1, 0) @ L @ Y).trace()

        # calculate L_2nd (Eq. 3)
        B = torch.where(X > 0, self.beta, 1)
        l_second = ((X - X_hat) * B).norm() ** 2

        # return weighted sum
        return l_second + self.alpha * l_first


class SDNE(torch.nn.Module):
    def __init__(self, num_nodes: int, hidden_sizes : tuple[int]=(256, 128)):
        """
        TODO move to a wrapper class that defines the training routine

        # check if graph is well-formed
        if graph.ndim != 2:
            raise ValueError('graph adjacency matrix must be two-dimensional')
        if graph.shape[0] != graph.shape[1]:
            raise ValueError('graph adjacency matrix must be a square matrix')
        """

        # call to super class
        super(SDNE, self).__init__()

        # set class variables
        self.num_nodes = num_nodes

        # set up encoder and decoder
        encoder_layers = [torch.nn.Linear(self.num_nodes, hidden_sizes[0])]
        decoder_layers = [torch.nn.Linear(hidden_sizes[0], self.num_nodes)]

        for m, n in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            # add activation function and new layer to encoder
            encoder_layers.append(torch.nn.Sigmoid())
            encoder_layers.append(torch.nn.Linear(m, n))

            # add activation function and new layer to the front of the decoder
            decoder_layers.insert(0, torch.nn.Sigmoid())
            decoder_layers.insert(0, torch.nn.Linear(n, m))

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)

        return reconstructed, embedding

    @torch.no_grad()
    def embed(self, x):
        return self.encoder(x)

############################################################################################
# for now, just spell out the training routine linearly; refactor into wrapper class later #
############################################################################################

# read in the graph
graph = read_network_file("clics4/edgelist-Family_Count.tsv")

# generate L matrix
D = np.diag(graph.sum(axis=1))
L = D - graph

# convert graph and L to torch tensors
graph = torch.tensor(graph, dtype=torch.float32)
L = torch.tensor(L, dtype=torch.float32)

# set up model
model = SDNE(num_nodes=graph.shape[0], hidden_sizes=(256, 128))

# use custom loss function
loss_function = SDNELoss(alpha=0.2, beta=10)

# the weight_decay is responsible for L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

losses = []
msl_losses = []

msl = msl_ratings()
for pair in set(msl.keys()):
    c1, c2 = pair
    if not (c1 in concept_to_id and c2 in concept_to_id):
        msl.pop(pair)

# early stopping
best_loss = np.inf
wait = 0
patience = 0

# for now, batched training is not implemented
for epoch in range(10000):
    model.train()
    reconstructed, embedding = model(graph)

    # calculate loss
    loss = loss_function(graph, reconstructed, embedding, L)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(float(loss))

    # MSL validation
    msl_similarities = []
    pred_similarities = []
    for pair, sim in msl.items():
        c1, c2 = pair
        idx1, idx2 = concept_to_id[c1], concept_to_id[c2]
        emb1 = model.embed(graph[idx1])
        emb2 = model.embed(graph[idx2])
        # cosine similarity between embeddings
        pred_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        msl_similarities.append(sim)
        pred_similarities.append(pred_sim)

    msl_corr = float(spearmanr(msl_similarities, pred_similarities).statistic)
    msl_losses.append(msl_corr)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Training loss: {loss.item():.4f}, MSL: {msl_corr:.4f}")

    # check for convergence
    if loss.item() < best_loss:
        best_loss = loss.item()
        wait = 0
    else:
        wait += 1
        if patience and wait > patience:
            print(f"Training stopped after {epoch} epochs.")
            break  # stop training


# Apply styling
plt.style.use('fivethirtyeight')

# Create the figure and the primary y-axis
fig, ax1 = plt.subplots()

# Plot training loss on the primary y-axis
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Training Loss', color='b')
ax1.plot(losses, color='b', label='Training Loss')
ax1.tick_params(axis='y', labelcolor='b')

# Create a secondary y-axis for MSL correlation
ax2 = ax1.twinx()
ax2.set_ylabel('MSL Correlation', color='g')
ax2.plot(msl_losses, color='g', label='MSL Correlation')
ax2.tick_params(axis='y', labelcolor='g')

# Add a title and legends
fig.suptitle('Training Loss and MSL Correlation')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
fig.tight_layout()
plt.show()


#####################################
# EXPORT TRAINED EMBEDDINGS TO FILE #
#####################################


with open("embeddings/2025-01-07/sdne.tsv", "w") as f:
    for concept, id in concept_to_id.items():
        embed = model.embed(graph[id])
        f.write(concept + "\t" + str(embed.detach().tolist()) + "\n")

