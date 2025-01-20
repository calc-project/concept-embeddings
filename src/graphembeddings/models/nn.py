import torch


class CBOW(torch.nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781

    Taken from https://github.com/OlgaChernytska/word2vec-pytorch/blob/main/utils/model.py (with slight modifications)
    """
    def __init__(self, vocab_size: int, embed_dimension: int = 128, padding=True):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=1,
            padding_idx=(vocab_size - 1) if padding else None
        )
        self.linear = torch.nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram(torch.nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781

    Taken from https://github.com/OlgaChernytska/word2vec-pytorch/blob/main/utils/model.py (with slight modifications)
    """
    def __init__(self, vocab_size: int, embed_dimension: int = 128):
        super(SkipGram, self).__init__()
        self.embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=1,
        )
        self.linear = torch.nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x


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


class SDNEEmbedder(torch.nn.Module):
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
        super(SDNEEmbedder, self).__init__()

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

