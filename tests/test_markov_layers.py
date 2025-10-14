import pytest
import torch
import torch_geometric.utils

from markov import EdgeProb, LinearFixedPoint


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_probs(small_network):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)

    fixed_point = LinearFixedPoint(node_dim=-1)
    edge_prob = EdgeProb(node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.type_as(rewards).unsqueeze(0)
    initial_values = sink_node_mask.clone()

    exp_values, _, _ = fixed_point(
        torch_graph.edge_index, rewards.exp(), sink_node_mask, initial_values, f_solver="fixed_point_iter", f_tol=1e-5
    )
    values = exp_values.log()
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()

    probs = edge_prob(torch_graph.edge_index, rewards.exp(), exp_values)
    assert torch.isclose(probs, torch_graph.prob, atol=1e-4).all()
