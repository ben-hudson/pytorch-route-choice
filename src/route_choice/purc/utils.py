import networkx as nx
import torch


def dense_incidence_matrix(
    edge_index: torch.Tensor,
    n_nodes: int = None,
    n_edges: int = None,
    dtype: torch.dtype = torch.float32,
):
    if n_nodes is None:
        n_nodes = torch.unique(edge_index).size(0)
    if n_edges:
        n_edges = edge_index.size(1)

    incidence = torch.zeros((n_nodes, n_edges), dtype=dtype)
    incidence.scatter_(0, edge_index[0].unsqueeze(0), -1)
    incidence.scatter_(0, edge_index[1].unsqueeze(0), 1)
    return incidence


def sparse_incidence_matrix(
    edge_index: torch.Tensor,
    n_nodes: int = None,
    n_edges: int = None,
    dtype: torch.dtype = torch.float32,
):
    if n_nodes is None:
        n_nodes = torch.unique(edge_index).size(0)
    if n_edges:
        n_edges = edge_index.size(1)

    edge_number = torch.arange(n_edges)

    tails = -torch.ones_like(edge_index[0])
    heads = torch.ones_like(edge_index[1])

    tail_coords = torch.stack((edge_index[0], edge_number))
    head_coords = torch.stack((edge_index[1], edge_number))

    indices = torch.cat((tail_coords, head_coords), dim=-1)
    values = torch.cat((tails, heads), dim=-1).to(dtype)

    return torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_edges))


def load_purc_toy_network():
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("d", pos=(1, 0))
    G.add_node("n", pos=(0.5, -0.5))

    G.add_edge("o", "d", link=1, length=2, rate=-1.0, flow=0.425)
    G.add_edge("o", "n", link=2, length=1, rate=-1.0, flow=0.575)
    G.add_edge("n", "d", link=3, length=1, rate=-1.0, flow=0.288)
    G.add_edge("n", "d", link=4, length=1, rate=-1.0, flow=0.288)
    G.add_edge("n", "o", link=5, length=1, rate=-1.0, flow=0.0)
    G.add_edge("o", "d", link=6, length=2, rate=-2.0, flow=0.0)

    return G
