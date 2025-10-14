import torch
import torchdeq

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class LinearFixedPoint(MessagePassing):
    """Solves a linear fixed-point problem x = Ax + b using message passing.

    This class implements a linear fixed-point solver using PyTorch Geometric's MessagePassing
    framework. It iteratively solves equations of the form x = Ax + b, where A is a sparse
    matrix represented via edge indices and values, and b is a vector.

    Args:
        **kwargs: Additional keyword arguments passed to MessagePassing base class.

    Methods:
        forward(edge_index, A, b, x0, **solver_kwargs):
            Solves the fixed-point problem using the specified solver.

            Args:
                edge_index (torch.Tensor): Edge indices of shape [2, num_edges] defining the sparse matrix structure
                A (torch.Tensor): Edge weights/values of shape [num_edges]
                b (torch.Tensor): Bias vector of shape [num_nodes]
                x0 (torch.Tensor): Initial guess for the solution of shape [num_nodes]
                **solver_kwargs: Additional arguments passed to the DEQ solver

            Returns:
                tuple: (solution, trajectory, solver_info) where:
                    - solution is the final fixed point x*
                    - trajectory is list of intermediate solutions
                    - solver_info contains solver statistics

    Examples:
        >>> solver = LinearFixedPoint()
        >>> x_star, traj, info = solver(edge_index, A, b, x0, solver='anderson', tol=1e-6)
    """

    def __init__(self, **kwargs):
        super().__init__(aggr="sum", flow="target_to_source", **kwargs)

    def forward(self, edge_index: torch.Tensor, A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor, **solver_kwargs):
        # some sensible defaults
        if "f_solver" not in solver_kwargs:
            solver_kwargs["f_solver"] = "fixed_point_iter"
        if "f_tol" not in solver_kwargs:
            solver_kwargs["f_tol"] = 1e-6

        solver = torchdeq.get_deq(**solver_kwargs)
        fixed_point = lambda x: self.propagate(edge_index, A=A, b=b, x=x)
        x_list, info = solver(fixed_point, x0)

        return x_list[-1], x_list, info

    def message(self, A: torch.Tensor, x_j: torch.Tensor):
        return A * x_j

    def update(self, Ax: torch.Tensor, b: torch.Tensor):
        return Ax + b


class EdgeProb(MessagePassing):
    """Compute edge probabilities in a graph using message passing.

    This class implements a numerically stable way to compute edge probabilities,
    avoiding the log-exp operations used in torch_geometric.utils.softmax.
    It uses target-to-source message passing to calculate probabilities based on
    edge rewards and node values.

    The probability of taking each edge is computed as:
        P(edge) = exp(Q(edge)) / sum(exp(Q(outgoing_edges)))
    where Q(edge) = reward(edge) + value(target_node)

    Args:
        **kwargs: Additional keyword arguments passed to MessagePassing base class

    Attributes:
        None

    Methods:
        forward(edge_index, exp_rewards, exp_values): Computes edge probabilities

    Example:
        >>> edge_prob = EdgeProb()
        >>> probs = edge_prob(edge_index, rewards.exp(), values.exp())
    """

    def __init__(self, **kwargs):
        super().__init__(aggr=None, flow="target_to_source", **kwargs)

    def forward(self, edge_index: torch.Tensor, exp_rewards: torch.Tensor, exp_values: torch.Tensor):
        prob = self.propagate(edge_index, exp_reward=exp_rewards, exp_value=exp_values)
        return prob

    def message(self, exp_reward: torch.Tensor, exp_value_j: torch.Tensor):
        exp_Q = exp_reward * exp_value_j
        return exp_Q

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        sum_over_edges = scatter(inputs, index, dim=self.node_dim, reduce="sum")
        prob = inputs / sum_over_edges.index_select(self.node_dim, index)
        return prob
