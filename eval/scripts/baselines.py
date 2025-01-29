import numpy as np
import networkx as nx
import warnings
from pathlib import Path

from graphembeddings.utils.postprocess import cosine_similarity
from graphembeddings.utils.io import read_graph_data
from graphembeddings.utils.graphutils import merge_graphs

warnings.filterwarnings("ignore")


class Baseline(object):
    def __init__(self, G, concept_to_ids):
        self.G = G
        self.G_inv = np.divide(1, G, where=G > 0)
        self.concept_to_ids = concept_to_ids
        self.ppmi_matrix = self._calculate_ppmi()
        self.random_walk_matrix = self._calculate_random_walk_matrix()
        self.nx_graph = nx.from_numpy_array(G, create_using=nx.DiGraph())

    def _calculate_ppmi(self):
        pmi = -np.log2((self.G / self.G.sum()) / (self.G.sum(axis=1).reshape(-1, 1) @ self.G.sum(axis=0).reshape(1, -1)))
        pmi[pmi < 0] = 0
        pmi[pmi == np.inf] = 0
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


GRAPHS_DIR = Path(__file__).parent.parent.parent / "data" / "graphs"

def get_all_graphs(graphs_dir=GRAPHS_DIR):
    graphs = {}
    concept_ids = {}

    # read in raw graphs...
    G_affix, _, concept_to_id_affix = read_graph_data(graphs_dir / "babyclics" / "affixfams.json", directed=True,
                                                      to_undirected=True)
    G_full, id_to_concept_full, concept_to_id_full = read_graph_data(graphs_dir / "babyclics" / "fullfams.json")
    G_overlap, id_to_concept_overlap, concept_to_id_overlap = read_graph_data(
        graphs_dir / "babyclics" / "overlapfams.json")
    graphs["affix"] = G_affix
    graphs["full"] = G_full
    graphs["overlap"] = G_overlap
    concept_ids["affix"] = concept_to_id_affix
    concept_ids["full"] = concept_to_id_full
    concept_ids["overlap"] = concept_to_id_overlap

    # ...and combine them
    G_full_affix, concepts_full_affix = merge_graphs(G_full, G_affix, concept_to_id_full, concept_to_id_affix)
    graphs["full+affix"] = G_full_affix
    concept_ids["full+affix"] = concepts_full_affix
    G_full_overlap, concepts_full_overlap = merge_graphs(G_full, G_overlap, concept_to_id_full, concept_to_id_overlap)
    graphs["full+overlap"] = G_full_overlap
    concept_ids["full+overlap"] = concepts_full_overlap
    G_all, concepts_all = merge_graphs(G_full_affix, G_overlap, concepts_full_affix, concept_to_id_overlap)
    graphs["full+affix+overlap"] = G_all
    concept_ids["full+affix+overlap"] = concepts_all

    return graphs, concept_ids
