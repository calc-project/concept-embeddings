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
    def __init__(self, vocab_size: int, embed_dimension: int = 128, bow_input_dim: int = None):
        super(SkipGram, self).__init__()
        if not bow_input_dim:
            self.embeddings = torch.nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dimension,
                max_norm=1,
            )
        else:
            self.embeddings = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=bow_input_dim,
                    out_features=embed_dimension,
                    bias=False,
                )
            )
        self.linear = torch.nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x


class NCELoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(NCELoss, self).__init__()

    def forward(self, pred, y):
        """
        Defines the loss function under negative sampling after Mikolov et al. (2013).
        :param y: the "true" labels with negative sampling. the actual node should be 1, the randomly sampled distractors -1, everything else 0.
        :param pred: the logits, where each index corresponds to a node.
        :return: the loss of the batch
        """
        return -torch.sum(torch.log(torch.sigmoid(y * pred))) / pred.size(0)
