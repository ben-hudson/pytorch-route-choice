import networkx as nx
import torch
import torch_geometric.utils

from torch.utils.data import TensorDataset
from typing import Any, Callable, Dict, List

from .models import MarkovRouteChoice


class SamplingException(BaseException):
    pass


class MarkovRouteChoiceDataset(TensorDataset):
    def __init__(
        self,
        nx_graph: nx.MultiDiGraph,
        edge_feats: List[str],
        demand: Dict[Any, float],
        dest: Any,
        reward_fn: Callable,
        n_samples: int,
    ):
        for n in nx_graph.nodes:
            nx_graph.nodes[n]["is_dest"] = n == dest
            nx_graph.nodes[n]["demand"] = demand.get(n, 0.0)

        for e in nx_graph.edges:
            nx_graph.edges[e]["reward"] = reward_fn(e)

        self.graph = torch_geometric.utils.from_networkx(nx_graph, group_edge_attrs=edge_feats)

        model = MarkovRouteChoice(None, -1)
        values, edge_probs = model.get_values_and_probs(self.graph.edge_index, self.rewards.exp(), self.sink_node_mask)
        node_flows, edge_flows = model.get_flows(self.graph.edge_index, edge_probs, self.demand)

        source_list = []
        path_list = []
        for _ in range(n_samples):
            # sample from demand distribution
            source_node = torch.multinomial(self.demand, 1).squeeze()
            source_node_mask = torch.nn.functional.one_hot(source_node, self.graph.num_nodes)
            # sample path
            path = self.sample_path(edge_probs, source_node_mask, self.graph.is_dest)

            source_list.append(source_node_mask)
            path_list.append(path)

        sources = torch.stack(source_list)
        paths = torch.stack(path_list)

        self.graph.value = values
        self.graph.edge_prob = edge_probs
        self.graph.node_flow = node_flows
        self.graph.edge_flow = edge_flows

        super().__init__(sources, paths)

    def sample_path(
        self,
        edge_probs: torch.Tensor,
        source_node_mask: torch.Tensor,
        sink_node_mask: torch.Tensor,
        max_length: int = 1000,
    ):
        chosen_edges = torch.zeros(self.graph.num_edges, dtype=torch.int64)
        current_state = torch.argmax(source_node_mask)

        sink_node_mask = sink_node_mask.bool()
        visited_states = [current_state]
        while not sink_node_mask[current_state] and len(visited_states) < max_length:
            outgoing_edges_mask = self.graph.edge_index[0] == current_state
            chosen_edge = torch.multinomial(edge_probs * outgoing_edges_mask, 1).squeeze()
            chosen_edge_mask = torch.nn.functional.one_hot(chosen_edge, self.graph.num_edges)

            chosen_edges |= chosen_edge_mask
            current_state = self.graph.edge_index[1, chosen_edge]
            visited_states.append(current_state)

        if not sink_node_mask[current_state]:
            raise SamplingException(f"Did not reach terminal state in {max_length} steps.")

        return chosen_edges

    @property
    def edge_index(self):
        return self.graph.edge_index

    @property
    def sink_node_mask(self):
        return self.graph.is_dest.unsqueeze(0)

    @property
    def rewards(self):
        return self.graph.reward.unsqueeze(0)

    @property
    def demand(self):
        return self.graph.demand.unsqueeze(0)


def load_rl_tutorial_network():
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("A", pos=(1, 0))
    G.add_node("B", pos=(2, 0))
    G.add_node("C", pos=(3, 0))
    G.add_node("D", pos=(4, 0))
    G.add_node("E", pos=(0, 1))
    G.add_node("F", pos=(1, 1))
    G.add_node("H", pos=(2, 1))
    G.add_node("I", pos=(3, 1))
    G.add_node("G", pos=(1, 2))
    G.add_node("d", pos=(4, 2))

    G.add_edge("o", "A", travel_time=0.3, flow=87.01)
    G.add_edge("A", "B", travel_time=0.1, flow=49.63)
    G.add_edge("B", "C", travel_time=0.1, flow=25.10)
    G.add_edge("C", "D", travel_time=0.3, flow=0.12)
    G.add_edge("o", "E", travel_time=0.4, flow=12.99)
    G.add_edge("A", "F", travel_time=0.1, flow=37.39)
    G.add_edge("B", "H", travel_time=0.2, flow=24.53)
    G.add_edge("C", "I", travel_time=0.1, flow=18.21)
    G.add_edge("C", "d", travel_time=0.9, flow=6.77)
    G.add_edge("D", "d", travel_time=2.6, flow=0.12)
    G.add_edge("E", "G", travel_time=0.3, flow=12.99)
    G.add_edge("F", "G", travel_time=0.3, flow=12.86)
    G.add_edge("F", "H", travel_time=0.2, flow=24.53)
    G.add_edge("H", "d", travel_time=0.5, flow=30.70)
    G.add_edge("H", "I", travel_time=0.2, flow=30.40)
    G.add_edge("I", "d", travel_time=0.3, flow=48.60)
    G.add_edge("G", "H", travel_time=0.6, flow=12.04)
    G.add_edge("G", "d", travel_time=0.7, flow=13.60)
    G.add_edge("G", "d", travel_time=2.8, flow=0.2)

    return G
