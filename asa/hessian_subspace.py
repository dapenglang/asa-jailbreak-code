"""
Hessian-Based Active Subspace Discovery
========================================
Computes the active subspace using the Hessian matrix of the adversarial
loss (second-order information) instead of the Fisher Information Matrix
(first-order gradient statistics).

The Hessian captures the precise curvature of the loss landscape:
    H = d^2 L / d x^2
while the empirical Fisher approximates:
    F ≈ E[g * g^T] ≈ E[H] (for log-likelihood losses)

Key difference:
    - AFIM (Fisher): gradient covariance -> variance directions
    - Hessian: exact curvature -> sharpness directions

Hessian-vector products (HVPs) are computed via Pearlmutter's trick:
    H @ v = d/dx [ (dL/dx)^T @ v ]

This requires only two backward passes (same cost as one gradient computation)
instead of the O(d^2) entries of the full Hessian.

Reference:
    Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian.
    Neural Computation, 6(2), 147-160.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HessianSubspace:
    """
    Hessian-based active subspace for adversarial loss landscape analysis.

    Uses Hessian-vector products (HVPs) with block Lanczos iteration to
    estimate the top-k eigenpairs of the Hessian matrix without ever forming
    the full Hessian explicitly.

    Attributes:
        model: The target language model.
        device (torch.device): Compute device.
        d (int): Dimensionality of the optimization variable.
        k (int): Number of top eigenpairs to compute.
        lanczos_iters (int): Number of Lanczos iteration steps.
    """

    def __init__(
        self,
        d: int,
        k: int = 32,
        lanczos_iters: int = 50,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the Hessian subspace analyzer.

        Args:
            d: Dimensionality of the suffix embedding space
               (suffix_length * embed_dim).
            k: Number of top eigenpairs to compute.
            lanczos_iters: Number of Lanczos iterations (default: 50).
            device: PyTorch device string.
        """
        if not (1 <= k <= d):
            raise ValueError(f"k must satisfy 1 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k
        self.lanczos_iters = lanczos_iters
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Hessian sketch buffer (HVP results accumulated over time)
        self.hvp_buffer: List[torch.Tensor] = []
        self.hessian_approx: Optional[torch.Tensor] = None

    @staticmethod
    def hessian_vector_product(
        loss: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product H @ v using Pearlmutter's trick.

        Pearlmutter's trick:
            H @ v = d/dx [ (dL/dx)^T @ v ]
                = d/dx [grad^T @ v]
                = gradient of (grad . v) w.r.t. x

        This requires two backward passes:
            1. Backward to get grad = dL/dx
            2. Compute scalar s = grad^T @ v, backward again to get H@v

        Args:
            loss: Scalar loss tensor.
            x: The variable w.r.t. which the Hessian is computed.
                Must have requires_grad=True.
            v: Direction vector of shape matching x.

        Returns:
            H @ v, same shape as v.
        """
        # First backward: get gradient
        grad = torch.autograd.grad(loss, x, create_graph=True)[0]

        # Compute the directional derivative: s = grad^T @ v
        s = torch.dot(grad.flatten(), v.flatten())

        # Second backward: gradient of s w.r.t. x gives H @ v
        hvp = torch.autograd.grad(s, x, retain_graph=False)[0]

        return hvp.detach()

    def update_hessian_approx(self, hvp_result: torch.Tensor) -> None:
        """
        Update the running Hessian approximation using HVP results.

        Accumulates rank-1 updates: H ≈ (1/N) * sum(v_i @ H @ v_i) * (v_i v_i^T)
        using the identity H ≈ sum_i (Hv_i / v_i) * v_i v_i^T (for unit v_i).

        In practice, we maintain a low-rank sketch by accumulating HVP
        results directly as outer products.

        Args:
            hvp_result: H @ v for some direction v, shape (d,).
        """
        self.hvp_buffer.append(hvp_result.detach().flatten())

        # Keep buffer manageable
        if len(self.hvp_buffer) > self.lanczos_iters:
            self.hvp_buffer.pop(0)

        if len(self.hvp_buffer) > 0:
            H = torch.stack(self.hvp_buffer, dim=0)  # (N, d)
            self.hessian_approx = (H.T @ H) / len(self.hvp_buffer)

    def lanczos(
        self,
        hvp_fn,
        k: Optional[int] = None,
        num_iters: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block Lanczos iteration for top-k eigenpairs via HVP function calls.

        Instead of forming the full Hessian, this uses the HVP function
        to implicitly apply H during Lanczos iteration.

        Args:
            hvp_fn: A callable that takes a vector v (shape (d,)) and
                    returns H @ v (shape (d,)). This avoids materializing H.
            k: Number of eigenpairs to compute (default: self.k).
            num_iters: Lanczos iterations (default: self.lanczos_iters).

        Returns:
            Tuple (eigenvalues, eigenvectors):
                - eigenvalues:  shape (k,), descending order.
                - eigenvectors: shape (d, k), orthonormal columns.
        """
        k = k or self.k
        num_iters = num_iters or self.lanczos_iters

        # Initialize with random orthonormal block
        Q = torch.randn(self.d, k, dtype=torch.float32, device=self.device)
        Q, _ = torch.linalg.qr(Q)

        # Tridiagonal matrix blocks for Rayleigh-Ritz
        T_blocks = []

        for i in range(num_iters):
            # Apply HVP: w = H @ Q[:, i] (or block version)
            if Q.shape[1] == k:
                w = torch.zeros_like(Q)
                for j in range(k):
                    w[:, j] = hvp_fn(Q[:, j])
            else:
                w = hvp_fn(Q[:, -1].flatten())
                w = w.unsqueeze(1)

            # Orthogonalize against all previous Q columns
            for j in range(Q.shape[1]):
                alpha = Q[:, j].t() @ w
                w = w - alpha * Q[:, j]

            beta = w.norm()
            if beta < 1e-10:
                break

            w = w / beta

            # Expand Q
            Q = torch.cat([Q, w.unsqueeze(1)], dim=1)

            # Keep Q manageable (truncate oldest columns)
            if Q.shape[1] > num_iters:
                Q = Q[:, 1:]

        # Rayleigh-Ritz: project onto Q subspace
        # B = Q^T H Q (approximate via HVP)
        B = torch.zeros(Q.shape[1], Q.shape[1], dtype=torch.float32, device=self.device)
        for i in range(Q.shape[1]):
            hvp_col = hvp_fn(Q[:, i].flatten())
            for j in range(i, Q.shape[1]):
                B[i, j] = Q[:, j] @ hvp_col
                B[j, i] = B[i, j]

        eigenvalues, eigenvectors_small = torch.linalg.eigh(B)
        idx = torch.argsort(eigenvalues, descending=True)

        eigenvalues = eigenvalues[idx][:k]
        eigenvectors = Q @ eigenvectors_small[:, idx][:, :k]

        return eigenvalues, eigenvectors

    def get_subspace_from_approx(
        self,
        k: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get top-k eigenpairs from the accumulated Hessian approximation.

        Falls back to full eigendecomposition of the approximate Hessian.
        For very large d, prefer using lanczos() with an HVP function instead.

        Args:
            k: Number of eigenpairs (default: self.k).

        Returns:
            Tuple (eigenvalues, eigenvectors) or None if insufficient data.
        """
        if self.hessian_approx is None:
            return None

        k = k or self.k
        k = min(k, self.d)

        eigenvalues, eigenvectors = torch.linalg.eigh(self.hessian_approx)
        idx = torch.argsort(eigenvalues, descending=True)

        return eigenvalues[idx][:k], eigenvectors[:, idx][:, :k]

    def compare_with_fisher(
        self,
        fisher_matrix: torch.Tensor,
        k: Optional[int] = None,
    ) -> dict:
        """
        Compare Hessian and Fisher subspaces for analysis.

        Computes:
            - Principal angle between top-k subspaces
            - Eigenvalue overlap ratio
            - Subspace agreement score

        Args:
            fisher_matrix: Fisher Information Matrix (d x d).
            k: Rank to compare (default: self.k).

        Returns:
            Dictionary with comparison metrics.
        """
        k = k or self.k
        result = {}

        if self.hessian_approx is None:
            result["status"] = "Hessian not yet estimated"
            return result

        # Top-k eigenvectors of each matrix
        h_eigvals, h_eigvecs = torch.linalg.eigh(self.hessian_approx)
        f_eigvals, f_eigvecs = torch.linalg.eigh(fisher_matrix.to(self.device))

        h_idx = torch.argsort(h_eigvals, descending=True)[:k]
        f_idx = torch.argsort(f_eigvals, descending=True)[:k]

        U_h = h_eigvecs[:, h_idx]  # (d, k)
        U_f = f_eigvecs[:, f_idx]  # (d, k)

        # Principal angles via SVD of U_h^T @ U_f
        overlap = U_h.t() @ U_f  # (k, k)
        _, s_vals, _ = torch.linalg.svd(overlap)

        result["principal_angles"] = torch.acos(s_vals.clamp(-1, 1)).cpu()
        result["max_alignment"] = s_vals[0].item()
        result["mean_alignment"] = s_vals.mean().item()
        result["subspace_agreement"] = (s_vals > 0.5).float().mean().item()

        return result
