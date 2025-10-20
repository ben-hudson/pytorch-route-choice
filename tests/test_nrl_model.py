import pytest
import torch
import torch_geometric.utils

from route_choice.markov.models.nested_recursive_logit import NestedRecursiveLogitRouteChoice


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_no_scales(small_network):
    for n in small_network.nodes:
        small_network.nodes[n]["is_dest"] = n == 4
    torch_graph = torch_geometric.utils.from_networkx(small_network)

    model = NestedRecursiveLogitRouteChoice(None, node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.unsqueeze(0)
    node_scales = torch.ones_like(sink_node_mask, dtype=torch.float32)
    values, probs = model.get_values_and_probs(
        torch_graph.edge_index,
        rewards,
        node_scales,
        sink_node_mask,
        f_solver="anderson",
        f_tol=1e-5,
    )
    assert torch.isclose(values, torch_graph.value, atol=1e-4).all()
    assert torch.isclose(probs, torch_graph.prob, atol=1e-4).all()


def test_nrl_model(nrl_toy_network):
    for n in nrl_toy_network.nodes:
        nrl_toy_network.nodes[n]["is_dest"] = n == "d"
    torch_graph = torch_geometric.utils.from_networkx(nrl_toy_network)

    model = NestedRecursiveLogitRouteChoice(None, node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    reward_scales = torch_graph.scale.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.unsqueeze(0)
    values, probs = model.get_values_and_probs(
        torch_graph.edge_index,
        rewards,
        reward_scales,
        sink_node_mask,
        f_solver="anderson",
        f_tol=1e-5,
    )
    probs = probs.squeeze(0)
    print(values, list(zip(torch_graph.name, probs)))

    paths = [("a", "a1"), ("a", "a2"), ("a", "a3"), ("b", "b1"), ("b", "b2"), ("b", "b3")]
    for path in paths:
        path_prob = 1.0
        for edge in path:
            edge_index = torch_graph.name.index(edge)
            path_prob *= probs[edge_index]
        print(path, path_prob)
