import torch
import torchdeq
import warnings

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, to_dense_adj

MIN_DENOMINATOR = 1e-30  # keeps the sink-node softmax division nonzero (finite fwd + bwd)


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
        try:
            x_list, info = self.solver(fixed_point, x0)
        except RuntimeError as e:
            # Degrade gracefully: return NaNs marked as not converged rather than crashing.
            warnings.warn(f"Solver failed with error: {e}")
            x = torch.full_like(x0, torch.nan)
            info = {"converged": x.new_zeros(x.size(0), dtype=torch.bool)}
            return x, info

        stop_mode = self.solver.f_stop_mode
        tolerance = self.solver.f_tol
        error = info[f"{stop_mode}_lowest"]
        converged = error <= tolerance
        if error.isnan().any():
            warnings.warn(
                f"Solver produced {error.isnan().sum()} NaN values. Check that inputs are finite and the system is well-conditioned."
            )
        elif not converged.all():
            warnings.warn(f"Solver did not converge: {stop_mode} error {error.max():.2e} > tol {tolerance}.")

        # nans show up as false in converged, so we don't need to add them
        info["converged"] = converged
        return x_list[-1], info

    def message(self, A: torch.Tensor, x_j: torch.Tensor):
        """Compute messages as element-wise product of edge values and source node states."""
        return A * x_j

    def update(self, Ax: torch.Tensor, b: torch.Tensor):
        """Add bias vector to aggregated messages to complete one fixed-point iteration."""
        return Ax + b


class DenseSolve(torch.nn.Module):
    """Dense linear solver for the fixed-point equation ``x = Ax + b``.

    Builds a dense transition matrix from sparse edge data and solves
    ``(I - A) * x = b`` using ``torch.linalg.solve``. This guarantees
    convergence regardless of the spectral radius of A and supports
    autograd differentiation.

    Drop-in replacement for :class:`LinearFixedPoint`.

    Args:
        node_dim: Dimension along which nodes are indexed. Must be ``-1``.
    """

    def __init__(self, node_dim: int = -1):
        super().__init__()
        if node_dim != -1:
            raise ValueError("DenseSolve only supports node_dim=-1.")

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_values: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
    ):
        """Solve ``(I - M) * x = b`` via dense linear solve.

        Args:
            edge_index: Edge indices of shape ``[2, num_edges]``.
            edge_values: Edge weights of shape ``[batch, num_edges]``.
            b: Right-hand side of shape ``[batch, num_nodes]``.
            x0: Initial guess (unused, accepted for interface compatibility).

        Returns:
            A tuple of ``(solution, info)`` where solution has shape
            ``[batch, num_nodes]`` and info is an empty dict.
        """
        batch_size, num_nodes = b.shape

        # to_dense_adj with [E, B] edge_attr produces [1, N, N, B]; move batch dim to front
        M = to_dense_adj(edge_index, edge_attr=edge_values.movedim(0, -1), max_num_nodes=num_nodes)
        M = M.squeeze(0).movedim(-1, 0)

        identity = torch.eye(num_nodes, device=M.device, dtype=M.dtype).unsqueeze(0)
        coefficient_matrix = identity - M
        rhs = b.type_as(edge_values).unsqueeze(-1)

        solution = torch.linalg.solve(coefficient_matrix, rhs).squeeze(-1)
        info = {"converged": torch.ones(batch_size, dtype=torch.bool, device=solution.device)}
        return solution, info


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
        # Clamp the denominator away from zero. A sink node's outgoing rewards are masked to 0 upstream,
        # so its outgoing exp_Q all vanish and sum_over_edges == 0 -- an unclamped division is then 0/0,
        # which produces a finite forward but a NaN backward (d prob / d sum = -exp_Q / sum**2 = 0 * inf)
        # that poisons any Jacobian taken through these probs. The clamp keeps the division finite in both
        # directions without perturbing legitimate O(1) sums.
        outgoing_sum = sum_over_edges.index_select(self.node_dim, index).clamp_min(MIN_DENOMINATOR)
        prob = exp_Q / outgoing_sum
        prob[is_sink_node_i.bool()] = 0.0  # sink nodes are absorbing: zero outgoing probability
        return prob
