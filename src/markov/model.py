import torch

from .layers import EdgeProb, LinearFixedPoint


class MarkovRouteChoice(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, node_dim: int = -1):
        super().__init__()

        self.node_dim = node_dim

        self.encoder = encoder
        self.fixed_point = LinearFixedPoint(node_dim=self.node_dim)
        self.edge_prob = EdgeProb(node_dim=self.node_dim)

    def forward(self, edge_index: torch.Tensor, edge_feats: torch.Tensor, sink_node_mask: torch.Tensor):
        exp_rewards = self.encoder(edge_feats)  # "logits"
        values, edge_probs = self.get_values_and_probs(edge_index, exp_rewards, sink_node_mask)

        return exp_rewards.log(), values, edge_probs

    def get_values_and_probs(self, edge_index, exp_rewards, sink_node_mask, **solver_kwargs):
        exp_values, _ = self.fixed_point(
            edge_index, exp_rewards, sink_node_mask, sink_node_mask.clone(), **solver_kwargs
        )
        assert (
            exp_values[sink_node_mask] == 1
        ), "Value at the terminal state is greater than zero. Maybe you forgot to remove outgoing edges?"

        edge_probs = self.edge_prob(edge_index, exp_rewards, exp_values)

        return exp_values.log(), edge_probs

    def get_flows(self, edge_index, edge_probs, demand, **solver_kwargs):
        node_flows, _ = self.fixed_point(edge_index.flip(0), edge_probs, demand, demand.clone(), **solver_kwargs)

        edge_flows = node_flows.index_select(self.node_dim, edge_index[0]) * edge_probs
        return node_flows, edge_flows
