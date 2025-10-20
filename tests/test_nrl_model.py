import itertools
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
    torch_graph = torch_geometric.utils.from_networkx(nrl_toy_network)

    model = NestedRecursiveLogitRouteChoice(None, node_dim=-1)

    rewards = -torch_graph.cost.unsqueeze(0)
    node_scales = torch_graph.scale.unsqueeze(0)
    sink_node_mask = torch_graph.is_dest.unsqueeze(0)
    values, probs = model.get_values_and_probs(
        torch_graph.edge_index,
        rewards,
        node_scales,
        sink_node_mask,
        f_solver="anderson",
        f_tol=1e-5,
    )
    probs = probs.squeeze(0)

    paths = [
        ("o", "a", "a1", "d"),
        ("o", "a", "a2", "d"),
        ("o", "a", "a3", "d"),
        ("o", "b", "b1", "d"),
        ("o", "b", "b2", "d"),
        ("o", "b", "b3", "d"),
    ]
    for path in paths:
        path_prob = 1.0
        for k, a in itertools.pairwise(path):
            edge_head_mask = torch_graph.edge_index[0] == torch_graph.name.index(k)
            edge_tail_mask = torch_graph.edge_index[1] == torch_graph.name.index(a)
            selected_edge = edge_head_mask & edge_tail_mask

            path_prob *= probs[selected_edge].item()
