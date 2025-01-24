import numpy as np
from sklearn.decomposition import PCA


def concatenate_embeddings(*embeddings):
    shared_keys = set(embeddings[0].keys())
    for i in range(1, len(embeddings)):
        shared_keys = shared_keys.intersection(embeddings[i].keys())

    concatenated_embeddings = {}

    for key in shared_keys:
        emb = embeddings[0][key].copy()
        for i in range(1, len(embeddings)):
            emb += embeddings[i][key]
        concatenated_embeddings[key] = emb

    return concatenated_embeddings


def fuse_embeddings(*input_embeddings, n_components=128, retain_all=False):
    concat_embeddings = concatenate_embeddings(*input_embeddings)
    keys, embeddings = zip(*concat_embeddings.items())
    embeddings = np.array(embeddings)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)

    embeddings = dict(zip(keys, embeddings))

    if retain_all:
        for input_embedding in input_embeddings:
            for k, v in input_embedding.items():
                if k not in embeddings and len(v) == n_components:
                    embeddings[k] = v

    return embeddings


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
