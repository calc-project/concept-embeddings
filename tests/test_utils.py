from conceptembeddings.utils.io import *
from conceptembeddings.utils.graphutils import *


def test_read_graph_directed(test_data, mock_concepts):
    graph_data1 = read_graph_data(test_data / "mockgraph_directed.json", directed=True, to_undirected=True)
    graph_data2 = read_graph_data(test_data / "mockgraph_directed.json", directed=True, to_undirected=False)
    assert graph_data1[1] == graph_data2[1]
    assert graph_data1[2] == graph_data2[2] == mock_concepts
    assert graph_data1[3] == graph_data2[3]
    assert (graph_data1[0] != graph_data2[0]).any()

def test_read_graph_undirected(test_data, mock_concepts):
    graph_data = read_graph_data(test_data / "mockgraph_undirected.json", directed=False)
    weights = graph_data[0]
    assert weights[0, 1] == weights[1, 0]

def test_read_embeddings(test_data):
    with_metadata = read_embeddings(test_data / "mockembeddings.json", metadata=True)
    without_metadata = read_embeddings(test_data / "mockembeddings.json", metadata=False)
    assert with_metadata["embeddings"] == without_metadata
    assert with_metadata.get("parameters")
    assert len(without_metadata["DUST"]) == 2

def test_merge_graphs(test_data):
    graph_data1 = read_graph_data(test_data / "mockgraph_directed.json", directed=True, to_undirected=True)
    graph_data2 = read_graph_data(test_data / "mockgraph_undirected.json", directed=False)
    merged_graph_data = merge_graphs(graph_data1[0], graph_data2[0], graph_data1[2], graph_data2[2])
    assert len(merged_graph_data[1]) == 3
    assert merged_graph_data[0][1, 0] == merged_graph_data[0][0, 1] == graph_data1[0][0, 1] + graph_data2[0][0, 2]
