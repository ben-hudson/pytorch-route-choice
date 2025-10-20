import torch
import torchdeq

from ..layers import LinearFixedPoint
from .recursive_logit import RLFixedPoint, RecursiveLogitRouteChoice


class NRLFixedPoint(RLFixedPoint):
    def forward(
        self,
        edge_index: torch.Tensor,
        exp_scaled_rewards: torch.Tensor,
        node_scales: torch.Tensor,
        sink_node_mask: torch.Tensor,
        z0: torch.Tensor,
        **solver_kwargs
    ):
        # this fixed-point problem is non-linear, so we shouldn't use the same defaults as the base class
        if "f_solver" not in solver_kwargs:
            solver_kwargs["f_solver"] = "anderson"
        if "f_tol" not in solver_kwargs:
            solver_kwargs["f_tol"] = 1e-6

        solver = torchdeq.get_deq(**solver_kwargs)
        fixed_point = lambda z: self.propagate(edge_index, M=exp_scaled_rewards, mu=node_scales, b=sink_node_mask, z=z)
        x_list, info = solver(fixed_point, z0.type_as(exp_scaled_rewards))

        return x_list[-1], info

    def message(self, M: torch.Tensor, mu_i: torch.Tensor, mu_j: torch.Tensor, z_j: torch.Tensor):
        X = z_j.pow(mu_j / mu_i)
        return M * X


class NestedRecursiveLogitRouteChoice(RecursiveLogitRouteChoice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_value = NRLFixedPoint(node_dim=kwargs.get("node_dim", -1))

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
        node_scales: torch.Tensor,
        sink_node_mask: torch.Tensor,
        **solver_kwargs
    ):

        edge_scales = node_scales.index_select(self.node_dim, edge_index[0])
        exp_scaled_rewards = (rewards / edge_scales).exp()
        z, _ = self.node_value(
            edge_index, exp_scaled_rewards, node_scales, sink_node_mask, sink_node_mask.clone(), **solver_kwargs
        )
        values = z.log() * node_scales

        # edge_probs = self.edge_prob(edge_index, exp_scaled_rewards, z)
        # According to the paper the express route to computing P is
        # TODO: apply this in the recursive logit model too
        edge_head_z = z.index_select(self.node_dim, edge_index[0])
        edge_tail_z = z.index_select(self.node_dim, edge_index[1])
        edge_probs = exp_scaled_rewards * edge_tail_z / edge_head_z

        return values, edge_probs
