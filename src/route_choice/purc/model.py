import torch


class PerturbedUtilityRouteChoice(torch.nn.Module):
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
        assert max <= 0, "max must be less or equal to zero (utilities are negative)"
        assert max > min, "max must be greater than min"

        rates = feats @ self.beta
        rates_std = (rates - rates.min()) / (rates.max() - rates.min())
        rates_scaled = rates_std * (max - min) + min

        return rates_scaled
