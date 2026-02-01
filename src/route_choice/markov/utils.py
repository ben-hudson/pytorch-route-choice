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
    M = scipy.sparse.coo_array((edge_values.numpy(), edge_index.numpy()), shape=shape).tocsr()
    A = scipy.sparse.eye(*M.shape) - M
    b = node_values.type_as(edge_values).numpy()
    assert (
        A.shape[0] == A.shape[1] == b.shape[0]
    ), f"Expected a square matrix and a vector but got {A.shape} and {b.shape}"
    if method == "spsolve":
        x = scipy.sparse.linalg.spsolve(A, b, **method_kwargs)
        info = None
    elif method == "gmres":
        x, info = scipy.sparse.linalg.gmres(A, b, **method_kwargs)
    else:
        raise ValueError(f"Expected 'spsolve' or 'gmres' but got {method}")
    return torch.tensor(x).type_as(edge_values), info
