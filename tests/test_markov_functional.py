import pytest
import torch
import torch_geometric.utils

from route_choice.markov import EdgeProb, spsolve


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
@pytest.mark.parametrize("method", ["spsolve", "gmres"])
def test_values_and_probs(small_network, method):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)
    shape = (torch_graph.num_nodes, torch_graph.num_nodes)

    exp_values, _ = spsolve(
        torch_graph.edge_index, (-torch_graph.cost).exp(), torch_graph.is_dest, shape, method=method
    )
    values = exp_values.log()
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()


@pytest.mark.parametrize("method", ["spsolve", "gmres"])
def test_flows(rl_tutorial_network, method):
    for n in rl_tutorial_network.nodes:
        rl_tutorial_network.nodes[n]["is_orig"] = n == "o"
        rl_tutorial_network.nodes[n]["is_dest"] = n == "d"
    torch_graph = torch_geometric.utils.from_networkx(rl_tutorial_network)
    shape = (torch_graph.num_nodes, torch_graph.num_nodes)

    edge_prob = EdgeProb(node_dim=-1)

    rewards = -2.0 * torch_graph.travel_time - 0.01
    sink_node_mask = torch_graph.is_dest.type_as(rewards)

    exp_values, _ = spsolve(torch_graph.edge_index, rewards.exp(), sink_node_mask, shape, method=method)
    probs = edge_prob(torch_graph.edge_index, rewards.exp(), exp_values, sink_node_mask)

    demand = torch_graph.is_orig.type_as(rewards) * 100
    # flip is transpose for COO matrices
    node_flows, _ = spsolve(torch_graph.edge_index.flip(0), probs, demand, shape, method=method)

    edge_flows = node_flows.index_select(-1, torch_graph.edge_index[0]) * probs
    assert torch.isclose(edge_flows, torch_graph.flow, atol=1e-2).all()
