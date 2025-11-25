import torch

from torch_geometric.utils import scatter

from ..layers import EdgeProb, LinearFixedPoint


class RLFixedPoint(LinearFixedPoint):
    def update(self, Ax: torch.Tensor, b: torch.Tensor):
        # In the recursive logit model b is a one-hot vector indicating the terminal state
        # There are  no edges leaving the terminal state, and its value is always 0 (exp(0) = 1).
        # To avoid modifying the underlying network, we can just override the value at the terminal state.
        Ax[b.bool()] = 1.0
        return Ax


class RecursiveLogitRouteChoice(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, node_dim: int = -1):
        super().__init__()

        self.node_dim = node_dim

        self.encoder = encoder
        self.node_value = RLFixedPoint(node_dim=self.node_dim)
        self.fixed_point = LinearFixedPoint(node_dim=self.node_dim)
        self.edge_prob = EdgeProb(node_dim=self.node_dim)

    def forward(
        self, edge_index: torch.Tensor, edge_feats: torch.Tensor, sink_node_mask: torch.Tensor, **solver_kwargs
    ):
        unscaled_rewards = self.encoder(edge_feats).squeeze(-1)

        # we need to scale exp(reward) such that the sum over each row is < 1
        # see: https://pubsonline.informs.org/doi/full/10.1287/trsc.2022.1145
        exp_unscaled_rewards = unscaled_rewards.exp()
        sum_over_rows = scatter(exp_unscaled_rewards, edge_index[0], dim=self.node_dim, reduce="sum")
        exp_rewards = exp_unscaled_rewards / sum_over_rows.max()

        values, edge_probs = self.get_values_and_probs(edge_index, exp_rewards, sink_node_mask, **solver_kwargs)
        return exp_rewards.log(), values, edge_probs

    def get_values_and_probs(
        self, edge_index: torch.Tensor, exp_rewards: torch.Tensor, sink_node_mask: torch.Tensor, **solver_kwargs
    ):

        exp_values, _ = self.node_value(
            edge_index,
            exp_rewards.unsqueeze(0),
            sink_node_mask.unsqueeze(0),
            sink_node_mask.clone().unsqueeze(0),
            **solver_kwargs
        )
        exp_values = exp_values.squeeze(0)

        edge_probs = self.edge_prob(edge_index, exp_rewards, exp_values, sink_node_mask)

        return exp_values.log(), edge_probs

    def get_flows(self, edge_index: torch.Tensor, edge_probs: torch.Tensor, demand: torch.Tensor, **solver_kwargs):
        node_flows, _ = self.fixed_point(
            edge_index.flip(0),
            edge_probs.unsqueeze(0),
            demand.unsqueeze(0),
            demand.clone().unsqueeze(0),
            **solver_kwargs
        )
        node_flows = node_flows.squeeze(0)

        edge_flows = node_flows.index_select(self.node_dim, edge_index[0]) * edge_probs
        return node_flows, edge_flows
