import pytest
import torch
import torch_geometric.utils

from route_choice.markov.layers import EdgeProb, LinearFixedPoint


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs(small_network):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)

    fixed_point = LinearFixedPoint(node_dim=-1)
    edge_prob = EdgeProb(node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.type_as(rewards).unsqueeze(0)

    exp_values, _ = fixed_point(
        torch_graph.edge_index,
        rewards.exp(),
        sink_node_mask,
        sink_node_mask.clone(),
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )
    values = exp_values.log()
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()

    probs = edge_prob(torch_graph.edge_index, rewards.exp(), exp_values, sink_node_mask)
    assert torch.isclose(probs, torch_graph.prob, atol=1e-4).all()


def test_flows(rl_tutorial_network):
    for n in rl_tutorial_network.nodes:
        rl_tutorial_network.nodes[n]["is_orig"] = n == "o"
        rl_tutorial_network.nodes[n]["is_dest"] = n == "d"
    torch_graph = torch_geometric.utils.from_networkx(rl_tutorial_network)

    fixed_point = LinearFixedPoint(node_dim=-1)
    edge_prob = EdgeProb(node_dim=-1)

    rewards = -2.0 * torch_graph.travel_time.unsqueeze(0) - 0.01
    sink_node_mask = torch_graph.is_dest.type_as(rewards).unsqueeze(0)

    exp_values, _ = fixed_point(
        torch_graph.edge_index,
        rewards.exp(),
        sink_node_mask,
        sink_node_mask.clone(),
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )
    probs = edge_prob(torch_graph.edge_index, rewards.exp(), exp_values, sink_node_mask)

    demand = torch_graph.is_orig.type_as(rewards).unsqueeze(0) * 100
    node_flows, _ = fixed_point(
        torch_graph.edge_index.flip(0),  # transpose for COO matrices
        probs,
        demand,
        demand.clone(),
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )

    edge_flows = node_flows.index_select(-1, torch_graph.edge_index[0]) * probs
    assert torch.isclose(edge_flows, torch_graph.flow, atol=1e-2).all()
