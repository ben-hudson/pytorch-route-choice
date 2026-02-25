import scipy.sparse
import scipy.sparse.linalg
import torch


def spsolve(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    node_values: torch.Tensor,
    shape=None,
    method="spsolve",
    **method_kwargs,
):
    """Solve the sparse linear system (I - M)x = b, where M is defined by edges.

    Constructs a sparse matrix M from the given edge index and values, then solves
    the system (I - M)x = b using SciPy sparse solvers. This is used as an
    alternative to iterative fixed-point solving for x = Mx + b.

    Args:
        edge_index: COO format edge indices of shape ``[2, num_edges]``.
        edge_values: Edge weights of shape ``[num_edges]``.
        node_values: Right-hand side vector b of shape ``[num_nodes]``.
        shape: Shape of the sparse matrix. Inferred from edge_index if None.
        method: Solver to use, either ``"spsolve"`` (direct) or ``"gmres"`` (iterative).
        **method_kwargs: Additional keyword arguments passed to the SciPy solver.

    Returns:
        A tuple of (solution, info) where solution is a tensor of shape ``[num_nodes]``
        and info is None for ``"spsolve"`` or the convergence status for ``"gmres"``.
    """
    assert not any([edge_values.requires_grad, node_values.requires_grad]), "Detach tensors before using spsolve."
    M = scipy.sparse.coo_array((edge_values.numpy(), edge_index.numpy()), shape=shape).tocsr()
    A = scipy.sparse.eye(*M.shape) - M
    b = node_values.type_as(edge_values).numpy()
    assert (
        A.shape[0] == A.shape[1] == b.shape[0]
    ), f"Expected a square matrix and a vector but got {A.shape} and {b.shape}."
    if method == "spsolve":
        x = scipy.sparse.linalg.spsolve(A, b, **method_kwargs)
        info = None
    elif method == "gmres":
        x, info = scipy.sparse.linalg.gmres(A, b, **method_kwargs)
    else:
        raise ValueError(f"Expected 'spsolve' or 'gmres' but got {method}.")
    return torch.tensor(x).type_as(edge_values), info
