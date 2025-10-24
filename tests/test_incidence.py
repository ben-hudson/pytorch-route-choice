import pytest
import networkx as nx
import torch_geometric.utils

from route_choice.purc.utils import dense_incidence_matrix, sparse_incidence_matrix


@pytest.mark.parametrize("n_nodes, p", [(10, 0.4), (100, 0.2), (1000, 0.1)])
def test_dense_incidence_matrix(n_nodes, p):
    nx_graph = nx.erdos_renyi_graph(n_nodes, p, directed=True)
    nx_incidence = nx.incidence_matrix(nx_graph, oriented=True).toarray()

    torch_graph = torch_geometric.utils.from_networkx(nx_graph)
    torch_incidence = dense_incidence_matrix(torch_graph.edge_index, torch_graph.num_nodes, torch_graph.num_edges)
    assert (nx_incidence == torch_incidence).all()


@pytest.mark.parametrize("n_nodes, p", [(10, 0.4), (100, 0.2), (1000, 0.1)])
def test_sparse_incidence_matrix(n_nodes, p):
    nx_graph = nx.erdos_renyi_graph(n_nodes, p, directed=True)
    nx_incidence = nx.incidence_matrix(nx_graph, oriented=True).toarray()

    torch_graph = torch_geometric.utils.from_networkx(nx_graph)
    torch_incidence = sparse_incidence_matrix(torch_graph.edge_index, torch_graph.num_nodes, torch_graph.num_edges)
    assert (nx_incidence == torch_incidence.to_dense()).all()
