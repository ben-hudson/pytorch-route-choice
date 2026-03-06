import pytest
import torch
import torch_geometric.utils

from route_choice.markov.layers import DenseSolve, EdgeProb, LinearFixedPoint


@pytest.fixture(params=["fixed_point", "direct"])
def linear_solver(request):
    if request.param == "fixed_point":
        return LinearFixedPoint(node_dim=-1)
    return DenseSolve(node_dim=-1)


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs(small_network, linear_solver):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)

    edge_prob = EdgeProb(node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.type_as(rewards).unsqueeze(0)

    exp_rewards = rewards.exp()
    leaves_sink_node = sink_node_mask.bool().index_select(-1, torch_graph.edge_index[0])
    exp_rewards = exp_rewards.masked_fill(leaves_sink_node, 0.0)

    exp_values, _ = linear_solver(torch_graph.edge_index, exp_rewards, sink_node_mask, sink_node_mask.clone())
    values = exp_values.log()
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()

    probs = edge_prob(torch_graph.edge_index, exp_rewards, exp_values, sink_node_mask)
    assert torch.isclose(probs, torch_graph.prob, atol=1e-4).all()


def test_flows(rl_tutorial_network, linear_solver):
    for n in rl_tutorial_network.nodes:
        rl_tutorial_network.nodes[n]["is_orig"] = n == "o"
        rl_tutorial_network.nodes[n]["is_dest"] = n == "d"
    torch_graph = torch_geometric.utils.from_networkx(rl_tutorial_network)

    edge_prob = EdgeProb(node_dim=-1)

    rewards = -2.0 * torch_graph.travel_time.unsqueeze(0) - 0.01
    sink_node_mask = torch_graph.is_dest.type_as(rewards).unsqueeze(0)

    exp_rewards = rewards.exp()
    leaves_sink_node = sink_node_mask.bool().index_select(-1, torch_graph.edge_index[0])
    exp_rewards = exp_rewards.masked_fill(leaves_sink_node, 0.0)

    exp_values, _ = linear_solver(torch_graph.edge_index, exp_rewards, sink_node_mask, sink_node_mask.clone())
    probs = edge_prob(torch_graph.edge_index, exp_rewards, exp_values, sink_node_mask)

    demand = torch_graph.is_orig.type_as(rewards).unsqueeze(0) * 100
    # flip is transpose for COO matrices
    node_flows, _ = linear_solver(torch_graph.edge_index.flip(0), probs, demand, demand.clone())

    edge_flows = node_flows.index_select(-1, torch_graph.edge_index[0]) * probs
    assert torch.isclose(edge_flows, torch_graph.flow, atol=1e-2).all()
