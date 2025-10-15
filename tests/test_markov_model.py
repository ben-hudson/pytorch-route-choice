import pytest
import torch
import torch_geometric.utils

from markov import MarkovRouteChoice


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs(small_network):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)

    model = MarkovRouteChoice(None, node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.unsqueeze(0)
    values, probs = model.get_values_and_probs(
        torch_graph.edge_index,
        rewards.exp(),
        sink_node_mask,
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()
    assert torch.isclose(probs, torch_graph.prob, atol=1e-4).all()


def test_flows(rl_tutorial_network):
    for n in rl_tutorial_network.nodes:
        rl_tutorial_network.nodes[n]["is_orig"] = n == "o"
        rl_tutorial_network.nodes[n]["is_dest"] = n == "d"
    torch_graph = torch_geometric.utils.from_networkx(rl_tutorial_network)

    model = MarkovRouteChoice(None, node_dim=-1)

    rewards = -2.0 * torch_graph.travel_time.unsqueeze(0) - 0.01
    sink_node_mask = torch_graph.is_dest.unsqueeze(0)
    values, probs = model.get_values_and_probs(
        torch_graph.edge_index,
        rewards.exp(),
        sink_node_mask,
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )

    demand = torch_graph.is_orig.type_as(probs).unsqueeze(0) * 100
    node_flows, edge_flows = model.get_flows(
        torch_graph.edge_index,
        probs,
        demand,
        f_solver="fixed_point_iter",
        f_tol=1e-5,
    )
    assert torch.isclose(edge_flows, torch_graph.flow, atol=1e-2).all()
