import torch
import torchdeq

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

        self.node_dim = node_dim

        self.encoder = encoder
        self.node_value = RLFixedPoint(node_dim=self.node_dim)
        self.fixed_point = LinearFixedPoint(node_dim=self.node_dim)
        self.edge_prob = EdgeProb(node_dim=self.node_dim)

        # some sensible defaults
        if "f_solver" not in solver_kwargs:
            solver_kwargs["f_solver"] = "anderson"
        if "f_tol" not in solver_kwargs:
            solver_kwargs["f_tol"] = 1e-6
        self.solver = torchdeq.get_deq(**solver_kwargs)

    def forward(self, edge_index: torch.Tensor, edge_feats: torch.Tensor, sink_node_mask: torch.Tensor):
        # this isn't very efficient since we do .exp() after
        rewards = -torch.nn.functional.softplus(self.encoder(edge_feats)).squeeze(-1)

        values, edge_probs = self.get_values_and_probs(edge_index, rewards.exp(), sink_node_mask)
        return rewards, values, edge_probs

    def get_values_and_probs(self, edge_index: torch.Tensor, exp_rewards: torch.Tensor, sink_node_mask: torch.Tensor):
        fixed_point = lambda z: self.node_value(edge_index, exp_rewards, sink_node_mask, z)
        exp_values_list, info = self.solver(fixed_point, sink_node_mask.clone().type_as(exp_rewards))
        exp_values = exp_values_list[-1]

        edge_probs = self.edge_prob(edge_index, exp_rewards, exp_values, sink_node_mask)

        return exp_values.log(), edge_probs

    def get_flows(self, edge_index: torch.Tensor, edge_probs: torch.Tensor, demand: torch.Tensor):
        fixed_point = lambda z: self.fixed_point(edge_index.flip(0), edge_probs, demand, z)
        node_flows_list, info = self.solver(fixed_point, demand.clone())
        node_flows = node_flows_list[-1]

        edge_flows = node_flows.index_select(self.node_dim, edge_index[0]) * edge_probs

        return node_flows, edge_flows
