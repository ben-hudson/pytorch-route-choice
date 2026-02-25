import torch
import torchdeq

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class LinearFixedPoint(MessagePassing):
    """Solve a linear fixed-point problem x = Ax + b using message passing.

    Uses PyTorch Geometric's MessagePassing framework to perform sparse
    matrix multiplicaition. The fixed-point solver from torchdeq enables
    implicit differentiation through the solution in the backward pass.

    Args:
        node_dim: Dimension along which nodes are indexed (negative offset to
            support batched operations). Defaults to ``-1``.
        **solver_kwargs: Keyword arguments passed to :func:`torchdeq.get_deq`.
            Defaults to ``f_solver="fixed_point_iter"`` and ``f_tol=1e-6`` if
            not specified.

    Example:
        >>> solver = LinearFixedPoint()
        >>> x_star, info = solver(A.indices(), A.values(), b, x0)
    """

    def __init__(self, node_dim: int = -1, **solver_kwargs):
        super().__init__(aggr="sum", flow="target_to_source", node_dim=node_dim)

        # some sensible defaults
        if "f_solver" not in solver_kwargs:
            solver_kwargs["f_solver"] = "fixed_point_iter"
        if "f_tol" not in solver_kwargs:
            solver_kwargs["f_tol"] = 1e-6
        self.solver = torchdeq.get_deq(**solver_kwargs)

    def forward(self, A_indices: torch.Tensor, A_values: torch.Tensor, b: torch.Tensor, x0: torch.Tensor):
        """Solve the fixed-point problem x = Ax + b.

        All inputs must include a batch dimension.

        Args:
            A_indices: Edge indices of shape ``[2, num_edges]``.
            A_values: Edge values of shape ``[batch, num_edges]``.
            b: Bias vector of shape ``[batch, num_nodes]``.
            x0: Initial guess of shape ``[batch, num_nodes]``.

        Returns:
            A tuple of (solution, info) where solution is the fixed point
            x* and info contains solver convergence statistics.
        """
        assert A_values.dim() == b.dim() == x0.dim(), f"Expected A, b, and x0 to have same dimensionality."
        assert A_values.dim() > 1, "Expected A, b, and x0 to have a batch dimension."

        b = b.type_as(A_values)
        x0 = x0.type_as(A_values)

        fixed_point = lambda x: self.propagate(A_indices, A=A_values, b=b, x=x)
        x_list, info = self.solver(fixed_point, x0)

        return x_list[-1], info

    def message(self, A: torch.Tensor, x_j: torch.Tensor):
        """Compute messages as element-wise product of edge values and source node states."""
        return A * x_j

    def update(self, Ax: torch.Tensor, b: torch.Tensor):
        """Add bias vector to aggregated messages to complete one fixed-point iteration."""
        return Ax + b


class EdgeProb(MessagePassing):
    """Compute edge transition probabilities using message passing.

    Computes the probability of traversing each edge in a graph, operating in
    the exponentiated domain for numerical stability (avoids log-exp round-trips
    used by :func:`torch_geometric.utils.softmax`).

    The probability of taking each edge is::

        P(edge) = exp(Q(edge)) / sum(exp(Q(outgoing_edges)))

    where ``Q(edge) = reward(edge) + value(target_node)``. Edges outgoing from
    sink nodes are assigned zero probability.

    Args:
        **kwargs: Keyword arguments passed to the ``MessagePassing`` base class.

    Example:
        >>> edge_prob = EdgeProb()
        >>> probs = edge_prob(edge_index, rewards.exp(), values.exp(), sink_mask)
    """

    def __init__(self, **kwargs):
        super().__init__(aggr=None, flow="target_to_source", **kwargs)

    def forward(
        self,
        edge_index: torch.Tensor,
        exp_rewards: torch.Tensor,
        exp_values: torch.Tensor,
        sink_node_mask: torch.Tensor,
    ):
        """Compute transition probabilities for all edges.

        Args:
            edge_index: Edge indices of shape ``[2, num_edges]``.
            exp_rewards: Exponentiated edge rewards of shape ``[num_edges]``.
            exp_values: Exponentiated node values of shape ``[num_nodes]``.
            sink_node_mask: Boolean mask indicating sink (absorbing) nodes.

        Returns:
            Edge probabilities of shape ``[num_edges]``.
        """
        prob = self.edge_updater(edge_index, exp_reward=exp_rewards, exp_value=exp_values, is_sink_node=sink_node_mask)
        return prob

    def edge_update(
        self,
        exp_reward: torch.Tensor,
        exp_value_j: torch.Tensor,
        is_sink_node_i: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor = None,
        dim_size: int = None,
    ):
        """Compute per-edge probabilities via softmax over outgoing edges.

        Normalizes ``exp(reward) * exp(value_target)`` over all outgoing edges
        of each source node, then zeros out edges from sink nodes.
        """
        exp_Q = exp_reward * exp_value_j
        sum_over_edges = scatter(exp_Q, index, dim=self.node_dim, reduce="sum")
        prob = exp_Q / sum_over_edges.index_select(self.node_dim, index)
        prob[is_sink_node_i.bool()] = 0.0  # gotcha
        return prob
