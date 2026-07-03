"""
Attack Fisher Information Matrix (AFIM)
========================================
Streaming estimation of the Fisher Information Matrix (FIM) from
attack gradients. The top eigenvectors of the FIM define the "active
subspace" in which adversarial updates are most effective.

Mathematical background:
    F_t = E[ g_t g_t^T ]  (outer product of gradients)
    We maintain a running average:
        F_{t+1} = (t/(t+1)) * F_t + (1/(t+1)) * g_t g_t^T
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class AttackFisherInformationMatrix:
    """
    Streaming estimator for the attack Fisher Information Matrix.

    The FIM captures the covariance structure of gradients observed
    during adversarial optimization. Its dominant eigenvectors span
    the "active subspace"---the low-dimensional manifold in which
    the loss landscape varies most sharply.

    Attributes:
        d (int): Embedding dimension.
        F (torch.Tensor): d x d running-average matrix.
        t (int): Number of updates observed so far.
        device (torch.device): Compute device.
    """

    def __init__(self, d: int, device: str = "cuda") -> None:
        """
        Initialize the AFIM.

        Args:
            d: Dimensionality of the gradient vectors (embedding dim).
            device: PyTorch device string.
        """
        self.d = d
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Initialize F as zero matrix
        self.F = torch.zeros((d, d), dtype=torch.float32, device=self.device)
        self.t = 0

    def update(self, g_t: torch.Tensor) -> None:
        """
        Update the running-average Fisher matrix with a new gradient.

        Formula:
            F_{t+1} = (t / (t+1)) * F_t + (1 / (t+1)) * (g_t g_t^T)

        Args:
            g_t: Gradient vector of shape (d,) or (d, 1).

        Raises:
            ValueError: If the gradient dimension does not match ``self.d``.
        """
        if g_t.dim() == 1:
            g_t = g_t.unsqueeze(1)  # (d, 1)
        elif g_t.dim() != 2 or g_t.shape[0] != self.d:
            raise ValueError(
                f"Expected gradient shape ({self.d},) or ({self.d}, 1), got {g_t.shape}"
            )

        self.t += 1
        # Rank-1 outer product update with decaying weight
        outer = g_t @ g_t.t()  # (d, d)
        self.F = ((self.t - 1) / self.t) * self.F + (1.0 / self.t) * outer

    def get_eigen_decomposition(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the top-k eigenvalues and eigenvectors of F.

        Uses ``torch.linalg.eigh`` (full decomposition) followed by
        selection of the largest-magnitude eigenpairs. For very large
        ``d`` consider switching to a Krylov method (e.g., LOBPCG).

        Args:
            k: Number of top eigenpairs to return.

        Returns:
            A tuple ``(eigenvalues, eigenvectors)`` where:
                - eigenvalues:  shape (k,), descending.
                - eigenvectors: shape (d, k), columns are eigenvectors.
        """
        if k > self.d:
            raise ValueError(f"k={k} cannot exceed dimension d={self.d}")

        # F is symmetric by construction, so eigh is numerically stable
        eigenvalues, eigenvectors = torch.linalg.eigh(self.F)

        # eigh returns ascending order; we want descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx][:k]
        eigenvectors = eigenvectors[:, idx][:, :k]

        return eigenvalues, eigenvectors

    def get_spectral_gap(self, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute the spectral gap ratio lambda_k / lambda_{k+1}.

        A large spectral gap indicates that a k-dimensional subspace
        captures most of the variance in the gradient distribution,
        justifying a low-rank approximation.

        Args:
            k: If provided, compute the gap at this specific rank.
               Otherwise use the current update count ``self.t`` clamped
               to ``[1, d-1]``.

        Returns:
            Scalar tensor containing the ratio. Returns inf if
            lambda_{k+1} is zero.
        """
        if k is None:
            k = max(1, min(self.t, self.d - 1))

        eigenvalues, _ = self.get_eigen_decomposition(k=k + 1)
        lambda_k = eigenvalues[k - 1]
        lambda_kp1 = eigenvalues[k]

        # Avoid division by zero
        if lambda_kp1.item() == 0.0:
            return torch.tensor(float("inf"), device=self.device)

        return lambda_k / lambda_kp1

    def estimate_power_law_beta(self, max_k: Optional[int] = None) -> torch.Tensor:
        """
        Estimate the power-law exponent beta such that
            lambda_i ~ C * i^{-beta}
        by fitting a linear model in log-log space.

        A steep power law (large beta) means energy is concentrated in
        very few dimensions, making low-rank subspace methods highly
        effective.

        Args:
            max_k: Number of eigenvalues to use for fitting. Defaults
                   to min(self.d, 256) to keep the regression cheap.

        Returns:
            Scalar tensor containing the fitted exponent beta (> 0).
        """
        if max_k is None:
            max_k = min(self.d, 256)

        eigenvalues, _ = self.get_eigen_decomposition(k=max_k)
        # Clamp to positive to avoid log(0)
        eigenvalues = eigenvalues.clamp(min=1e-12)

        # Indices: 1, 2, ..., max_k
        i = torch.arange(1, max_k + 1, dtype=torch.float32, device=self.device)

        # Linear regression in log-log space: log(lambda) = log(C) - beta * log(i)
        log_lambda = torch.log(eigenvalues)
        log_i = torch.log(i)

        # Ordinary least squares for slope
        mean_x = log_i.mean()
        mean_y = log_lambda.mean()
        numerator = ((log_i - mean_x) * (log_lambda - mean_y)).sum()
        denominator = ((log_i - mean_x) ** 2).sum()

        # beta = -slope because lambda ~ C * i^{-beta}
        beta = -numerator / denominator
        return beta.clamp(min=0.0)

    def effective_rank(self) -> torch.Tensor:
        """
        Compute the effective rank of the Fisher matrix.

        Definition (Roy & Vetterli):
            erank(F) = (sum_i lambda_i)^2 / (sum_i lambda_i^2)

        This quantity is always in [1, d]. A small effective rank
        relative to d indicates that the gradient covariance is
        intrinsically low-dimensional.

        Returns:
            Scalar tensor with the effective rank.
        """
        # We need all eigenvalues for this metric
        eigenvalues, _ = self.get_eigen_decomposition(k=self.d)
        trace = eigenvalues.sum()
        frob_sq = (eigenvalues ** 2).sum()

        if frob_sq.item() == 0.0:
            return torch.tensor(0.0, device=self.device)

        return (trace ** 2) / frob_sq
