import torch
import torchdeq

from .recursive_logit import RLFixedPoint


class NRLFixedPoint(RLFixedPoint):
    def forward(
        self,
        reward_indices: torch.Tensor,
        reward_values: torch.Tensor,
        reward_scales: torch.Tensor,
        sink_node_mask: torch.Tensor,
        x0: torch.Tensor,
        **solver_kwargs
    ):
        # this fixed-point problem is non-linear, so we shouldn't use the same defaults as the base class
        if "f_solver" not in solver_kwargs:
            solver_kwargs["f_solver"] = "anderson"
        if "f_tol" not in solver_kwargs:
            solver_kwargs["f_tol"] = 1e-6

        solver = torchdeq.get_deq(**solver_kwargs)
        fixed_point = lambda x: self.propagate(reward_indices, r=reward_values, mu=reward_scales, b=sink_node_mask, x=x)
        x_list, info = solver(fixed_point, x0.type_as(reward_values))

        return x_list[-1], info

    def message(self, r: torch.Tensor, mu_i: torch.Tensor, mu_j: torch.Tensor, x_j: torch.Tensor):
        M = (r / mu_i).exp()
        z = x_j.pow(mu_j / mu_i)
        return M * z


class NestedRecursiveLogitRouteChoice(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_value = NRLFixedPoint(node_dim=self.node_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        node_scales: torch.Tensor,
        sink_node_mask: torch.Tensor,
    ):
        rewards = -torch.nn.functional.softplus(self.encoder(edge_feats)).squeeze(-1)

        values, edge_probs = self.get_values_and_probs(edge_index, rewards, node_scales, sink_node_mask)
        return rewards, values, edge_probs

    def get_values_and_probs(
        self,
        edge_index: torch.Tensor,
        rewards: torch.Tensor,
        reward_scales: torch.Tensor,
        sink_node_mask: torch.Tensor,
        **solver_kwargs
    ):
        exp_values, _ = self.node_value(
            edge_index, rewards, reward_scales, sink_node_mask, sink_node_mask.clone(), **solver_kwargs
        )

        edge_probs = self.edge_prob(edge_index, rewards, exp_values)

        return exp_values.log(), edge_probs
