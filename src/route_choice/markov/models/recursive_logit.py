import torch

from ..layers import EdgeProb, LinearFixedPoint


class RLFixedPoint(LinearFixedPoint):
    def update(self, Ax: torch.Tensor, b: torch.Tensor):
        # In the recursive logit model b is a one-hot vector indicating the terminal state
        # There are  no edges leaving the terminal state, and its value is always 0 (exp(0) = 1).
        # To avoid modifying the underlying network, we can just override the value at the terminal state.
        Ax[b.bool()] = 1.0
        return Ax


class RecursiveLogitRouteChoice(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, node_dim: int = -1, **solver_kwargs):
        super().__init__()

        assert node_dim < 0, "node_dim must be specified as a negative offset."
        self.node_dim = node_dim

        self.encoder = encoder
        self.node_value = RLFixedPoint(node_dim=self.node_dim, **solver_kwargs)
        self.node_flow = LinearFixedPoint(node_dim=self.node_dim, **solver_kwargs)
        self.edge_prob = EdgeProb(node_dim=self.node_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        sink_node_mask: torch.Tensor,
        node_demand: torch.Tensor = None,
    ):
        rewards = self.encoder(edge_feats).squeeze(-1)
        values, edge_probs = self.get_values_and_probs(edge_index, rewards, sink_node_mask)
        if node_demand is None:
            return rewards, values, edge_probs
        else:
            node_flows, edge_flows = self.get_flows(edge_index, edge_probs, node_demand)
            return rewards, values, edge_probs, node_flows, edge_flows

    def get_values_and_probs(self, edge_index: torch.Tensor, rewards: torch.Tensor, sink_node_mask: torch.Tensor):
        assert (
            rewards.dim() + self.node_dim > 0
        ), "edge_rewards requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."
        assert (
            sink_node_mask.dim() + self.node_dim > 0
        ), "sink_node_mask requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."

        # scaling exp(reward) such that the sum over each row is < 1 guarantees convergence
        # see: https://pubsonline.informs.org/doi/full/10.1287/trsc.2022.1145
        # however, it can also cause numerical issues depending on how f_tol is set
        # TODO: not passing tests.
        exp_rewards = rewards.exp()

        exp_values, _ = self.node_value(edge_index, exp_rewards, sink_node_mask, sink_node_mask.clone())
        edge_probs = self.edge_prob(edge_index, exp_rewards, exp_values, sink_node_mask)
        return exp_values.log(), edge_probs

    def get_flows(self, edge_index: torch.Tensor, edge_probs: torch.Tensor, demand: torch.Tensor):
        assert (
            edge_probs.dim() + self.node_dim > 0
        ), "exp_rewards requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."
        assert (
            demand.dim() + self.node_dim > 0
        ), "sink_node_mask requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."

        node_flows, _ = self.node_flow(edge_index.flip(0), edge_probs, demand, demand.clone())
        edge_flows = node_flows.index_select(self.node_dim, edge_index[0]) * edge_probs
        return node_flows, edge_flows
