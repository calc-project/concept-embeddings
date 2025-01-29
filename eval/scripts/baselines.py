import numpy as np
import networkx as nx


from graphembeddings.utils.postprocess import cosine_similarity


class Baseline(object):
    def __init__(self, G, concept_to_ids):
        self.G = G
        self.G_inv = np.divide(1, G, where=G > 0)
        self.concept_to_ids = concept_to_ids
        self.ppmi_matrix = self._calculate_ppmi()
        self.random_walk_matrix = self._calculate_random_walk_matrix()
        self.nx_graph = nx.from_numpy_array(G, create_using=nx.DiGraph())

    def _calculate_ppmi(self):
        pmi = (self.G / self.G.sum()) / (self.G.sum(axis=1).reshape(-1, 1) @ self.G.sum(axis=0).reshape(1, -1))
        pmi[pmi < 0] = 0
        return pmi

    def _calculate_random_walk_matrix(self, alpha=0.75):
        p = self.ppmi_matrix * self.ppmi_matrix.sum()
        walk_matrix = (np.identity(len(self.G)) - alpha * p) ** -1
        walk_matrix[walk_matrix == np.inf] = 0
        return walk_matrix

    def shortest_path(self, c1, c2):
        id1 = self.concept_to_ids.get(c1)
        id2 = self.concept_to_ids.get(c2)

        if id1 in self.nx_graph and id2 in self.nx_graph and nx.has_path(self.nx_graph, id1, id2):
            return nx.shortest_path_length(self.nx_graph, id1, id2)

        return np.nan

    def ppmi(self, c1, c2):
        id1 = self.concept_to_ids.get(c1)
        id2 = self.concept_to_ids.get(c2)

        # check if both indices are valid and in bounds
        if id1 is not None and id2 is not None and 0 <= id1 < len(self.G) and 0 <= id2 < len(self.G):
            return self.ppmi_matrix[id1, id2]

        return np.nan

    def cos_similarity(self, c1, c2):
        id1 = self.concept_to_ids.get(c1)
        id2 = self.concept_to_ids.get(c2)

        # check if both indices are valid and in bounds
        if id1 is not None and id2 is not None and 0 <= id1 < len(self.G) and 0 <= id2 < len(self.G):
            return cosine_similarity(self.G[id1], self.G[id2])

        return np.nan

    def katz_random_walks(self, c1, c2):
        id1 = self.concept_to_ids.get(c1)
        id2 = self.concept_to_ids.get(c2)

        # check if both indices are valid and in bounds
        if id1 is not None and id2 is not None and 0 <= id1 < len(self.G) and 0 <= id2 < len(self.G):
            return cosine_similarity(self.random_walk_matrix[id1], self.random_walk_matrix[id2])

        return np.nan
