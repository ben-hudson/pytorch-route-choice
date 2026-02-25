import torch


class PerturbedUtilityRouteChoice(torch.nn.Module):
    """Perturbed utility route choice model (Fosgerau et al., 2022).

    Learns utility rates from observed edge flows by projecting out the
    equilibrium constraints via the network incidence matrix. The model solves
    a least-squares problem in the nullspace of the flow conservation
    constraints.

    Args:
        n_feats: Number of edge features.
        regularizer: Regularization function applied to flows.
            - ``"entropy"``: ``(1 + x) * ln(1 + x) - x``
            - ``"square"`` / ``"l2"``: ``x^2``
    """

    def __init__(self, n_feats: int, regularizer: str = "entropy"):
        super().__init__()

        self.beta = torch.nn.Parameter(torch.ones((n_feats, 1), dtype=torch.float32))

        self.reg = regularizer
        if self.reg == "entropy":
            self.reg_prime = lambda x: torch.log(1 + x)
        elif self.reg == "square" or self.reg == "l2":
            self.reg_prime = lambda x: x
        else:
            raise ValueError(f"Unknown value for regularizer: {self.reg}")

    def forward(
        self, incidence_matrix: torch.Tensor, edge_lengths: torch.Tensor, feats: torch.Tensor, flows: torch.Tensor
    ):
        """Compute projected residuals and least-squares loss.

        Projects the regularized cost and feature vectors into the nullspace of
        the incidence matrix (zeroing out flow conservation directions), then
        computes residuals between projected costs and projected features
        weighted by the learned utility rates.

        Args:
            incidence_matrix: Node-edge incidence matrix of shape
                ``[num_nodes, num_edges]``.
            edge_lengths: Edge lengths of shape ``[num_edges]``.
            feats: Edge features of shape ``[batch, num_edges, num_features]``.
            flows: Observed edge flows of shape ``[batch, num_edges]``.

        Returns:
            A tuple of (residuals, loss) where residuals has shape
            ``[batch, num_edges]`` and loss is a scalar sum of squared
            residuals.
        """
        batch_size, n_edges = flows.shape

        A = incidence_matrix.expand(batch_size, *incidence_matrix.shape)

        I = torch.eye(n_edges).expand(batch_size, n_edges, n_edges).to(A.device)

        zero_flow_mask = torch.isclose(flows, torch.tensor(0.0))
        B = I.clone()
        B_diag = torch.diagonal(B, dim1=1, dim2=2)
        B_diag[zero_flow_mask] = 0

        BA_T = B @ A.transpose(1, 2)
        C = torch.linalg.pinv(BA_T)
        P = (I - BA_T @ C) @ B

        edge_lengths_ex = edge_lengths[None, :, None]
        y = P @ (edge_lengths_ex * self.reg_prime(flows).unsqueeze(-1))
        w = P @ (edge_lengths_ex * feats)

        residuals = (y - (w @ self.beta)).squeeze(-1)
        loss = residuals.pow(2).sum()
        return residuals, loss

    def util_rate(self, feats: torch.Tensor, min: float = -1, max: float = 0):
        """Compute utility rates from edge features, scaled to a given range.

        Applies the learned weight vector to the features and rescales the
        result to lie within ``[min, max]`` via min-max normalization.

        Args:
            feats: Edge features of shape ``[num_edges, num_features]``.
            min: Lower bound of the output range. Defaults to ``-1``.
            max: Upper bound of the output range. Must be ``<= 0`` (utilities
                are negative). Defaults to ``0``.

        Returns:
            Scaled utility rates of shape ``[num_edges, 1]``.
        """
        assert max <= 0, "max must be less or equal to zero (utilities are negative)"
        assert max > min, "max must be greater than min"

        rates = feats @ self.beta
        rates_std = (rates - rates.min()) / (rates.max() - rates.min())
        rates_scaled = rates_std * (max - min) + min

        return rates_scaled
