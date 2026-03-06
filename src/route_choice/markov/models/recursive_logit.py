import torch

from ..layers import DenseSolve, EdgeProb, LinearFixedPoint


class RecursiveLogitRouteChoice(torch.nn.Module):
    """Recursive logit route choice model (Fosgerau et al., 2013).

    Computes node values, edge transition probabilities, and (optionally) edge
    flows on a directed graph. An encoder module maps edge features to scalar
    rewards, which are then used to solve a Bellman-like value function.

    Args:
        encoder: Module that maps edge features to scalar rewards.
        node_dim: Dimension along which nodes are indexed (negative offset to
            support batched operations). Defaults to ``-1``.
        solver: Method to solve the linear systems. ``"fixed_point"``
            uses iterative fixed-point solving (default). ``"direct"`` uses
            a dense linear solve via ``torch.linalg.solve``.
        **solver_kwargs: Keyword arguments passed to the fixed-point solvers.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        node_dim: int = -1,
        solver: str = "fixed_point",
        **solver_kwargs,
    ):
        super().__init__()

        assert node_dim < 0, "node_dim must be specified as a negative offset."
        self.node_dim = node_dim

        self.encoder = encoder
        if solver == "direct":
            self.linear_solver = DenseSolve(node_dim=self.node_dim)
        elif solver == "fixed_point":
            self.linear_solver = LinearFixedPoint(node_dim=self.node_dim, **solver_kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver!r}. Expected 'fixed_point' or 'direct'.")
        self.edge_prob = EdgeProb(node_dim=self.node_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        sink_node_mask: torch.Tensor,
        node_demand: torch.Tensor = None,
    ):
        """Compute rewards, values, probabilities, and optionally flows.

        Args:
            edge_index: Edge indices of shape ``[2, num_edges]``.
            edge_feats: Edge features of shape ``[batch, num_edges, num_features]``.
            sink_node_mask: Boolean mask of shape ``[batch, num_nodes]`` indicating
                sink (absorbing) nodes.
            node_demand: Optional demand vector of shape ``[batch, num_nodes]``. If
                provided, node and edge flows are also computed and returned.

        Returns:
            ``(rewards, values, edge_probs)`` if ``node_demand`` is None,
            otherwise ``(rewards, values, edge_probs, node_flows, edge_flows)``.
        """
        # Flatten batch and edge dims in case encoder has batch norm
        rewards = self.encoder(edge_feats.flatten(end_dim=-2)).reshape(edge_feats.shape[:-1])
        values, edge_probs = self.get_values_and_probs(edge_index, rewards, sink_node_mask)
        if node_demand is None:
            return rewards, values, edge_probs
        else:
            node_flows, edge_flows = self.get_flows(edge_index, edge_probs, node_demand)
            return rewards, values, edge_probs, node_flows, edge_flows

    def get_values_and_probs(self, edge_index: torch.Tensor, rewards: torch.Tensor, sink_node_mask: torch.Tensor):
        """Compute node values and edge transition probabilities from rewards.

        Solves the recursive logit value function in the exponentiated domain
        via fixed-point iteration, then derives per-edge transition probabilities.

        Args:
            edge_index: Edge indices of shape ``[2, num_edges]``.
            rewards: Edge rewards of shape ``[batch, num_edges]``.
            sink_node_mask: Boolean mask of shape ``[batch, num_nodes]`` indicating
                sink (absorbing) nodes.

        Returns:
            A tuple of (values, edge_probs) where values has shape
            ``[batch, num_nodes]`` (in log-space) and edge_probs has shape
            ``[batch, num_edges]``.
        """
        assert (
            rewards.dim() + self.node_dim > 0
        ), "edge_rewards requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."
        assert (
            sink_node_mask.dim() + self.node_dim > 0
        ), "sink_node_mask requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."

        exp_rewards = rewards.exp()
        # There is no reward for leaving the sink node as it is the terminal state.
        leaves_sink_node = sink_node_mask.bool().index_select(-1, edge_index[0])
        exp_rewards = exp_rewards.masked_fill(leaves_sink_node, 0.0)
        exp_values, _ = self.linear_solver(edge_index, exp_rewards, sink_node_mask, sink_node_mask.clone())
        edge_probs = self.edge_prob(edge_index, exp_rewards, exp_values, sink_node_mask)
        return exp_values.log(), edge_probs

    def get_flows(self, edge_index: torch.Tensor, edge_probs: torch.Tensor, demand: torch.Tensor):
        """Compute node and edge flows from transition probabilities and demand.

        Propagates demand through the network using the transition probabilities
        on the reversed graph to obtain node flows, then computes edge flows as
        the product of source node flow and edge probability.

        Args:
            edge_index: Edge indices of shape ``[2, num_edges]``.
            edge_probs: Edge transition probabilities of shape ``[batch, num_edges]``.
            demand: Node demand vector of shape ``[batch, num_nodes]``.

        Returns:
            A tuple of (node_flows, edge_flows) where node_flows has shape
            ``[batch, num_nodes]`` and edge_flows has shape ``[batch, num_edges]``.
        """
        assert (
            edge_probs.dim() + self.node_dim > 0
        ), "exp_rewards requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."
        assert (
            demand.dim() + self.node_dim > 0
        ), "sink_node_mask requires a batch dim, even if it is 1-dimensional! Use .unsqueeze(0)."

        node_flows, _ = self.linear_solver(edge_index.flip(0), edge_probs, demand, demand.clone())
        edge_flows = node_flows.index_select(self.node_dim, edge_index[0]) * edge_probs
        return node_flows, edge_flows
