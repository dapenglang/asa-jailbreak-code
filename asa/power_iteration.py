"""
Block Power Iteration for Efficient Top-k Eigendecomposition
============================================================
Computes the top-k eigenvalues and eigenvectors of a symmetric
positive semi-definite matrix using block (subspace) power iteration.

Complexity: O(d^2 * k * num_iters) instead of O(d^3) for full eigh.
Useful when d (embedding dimension) is large and only a small
subspace (k << d) is needed.

Reference:
    Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
"""

from typing import Optional, Tuple

import torch


class BlockPowerIteration:
    """
    Block power iteration for top-k eigenpairs of a symmetric PSD matrix.

    Iteratively refines a block of k orthonormal vectors to converge to
    the dominant eigenspace. At each iteration:
        1. Multiply: Y = A @ V
        2. Orthogonalize: QR decomposition of Y
        3. Extract eigenvalues via Rayleigh quotient: R = V^T @ A @ V

    Attributes:
        d (int): Matrix dimension.
        k (int): Number of top eigenpairs to compute.
        num_iters (int): Maximum number of power iterations.
        device (torch.device): Compute device.
    """

    def __init__(
        self,
        d: int,
        k: int,
        num_iters: int = 30,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the block power iteration solver.

        Args:
            d: Dimension of the symmetric matrix.
            k: Number of top eigenpairs to compute (k <= d).
            num_iters: Maximum number of iterations (default: 30).
            device: PyTorch device string.
        """
        if not (1 <= k <= d):
            raise ValueError(f"k must satisfy 1 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k
        self.num_iters = num_iters
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute(
        self,
        A: torch.Tensor,
        k: Optional[int] = None,
        num_iters: Optional[int] = None,
        tol: float = 1e-8,
        V_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k eigenpairs of symmetric PSD matrix A.

        Args:
            A: Symmetric PSD matrix of shape (d, d).
            k: Override number of eigenpairs (default: self.k).
            num_iters: Override iteration count (default: self.num_iters).
            tol: Convergence tolerance for eigenvalue change.
            V_init: Optional initial basis of shape (d, k) for warm-starting.

        Returns:
            Tuple (eigenvalues, eigenvectors):
                - eigenvalues:  shape (k,), descending order.
                - eigenvectors: shape (d, k), orthonormal columns.
        """
        if A.shape != (self.d, self.d):
            raise ValueError(f"Expected A shape ({self.d}, {self.d}), got {A.shape}")

        k = k or self.k
        num_iters = num_iters or self.num_iters
        A = A.to(self.device).float()

        # Initialize basis: random or user-provided
        if V_init is not None:
            V = V_init.to(self.device).float()
            if V.shape != (self.d, k):
                raise ValueError(f"V_init shape must be ({self.d}, {k}), got {V.shape}")
        else:
            V = torch.randn(self.d, k, dtype=torch.float32, device=self.device)
            V, _ = torch.linalg.qr(V)

        prev_eigenvalues = None

        for iteration in range(num_iters):
            # Step 1: Multiply A @ V
            Y = A @ V  # (d, k)

            # Step 2: QR orthogonalization
            V, _ = torch.linalg.qr(Y)

            # Step 3: Rayleigh quotient for eigenvalue estimates
            # R = V^T @ A @ V (k x k), then eigendecompose the small matrix
            R = V.t() @ A @ V  # (k, k)
            eigvals_small, eigvecs_small = torch.linalg.eigh(R)

            # Sort descending
            idx = torch.argsort(eigvals_small, descending=True)
            eigvals_small = eigvals_small[idx]
            eigvecs_small = eigvecs_small[:, idx]

            # Rotate basis to align with eigenvectors of Rayleigh quotient
            V = V @ eigvecs_small  # (d, k)

            # Convergence check
            if prev_eigenvalues is not None:
                delta = (eigvals_small - prev_eigenvalues).abs().max().item()
                if delta < tol:
                    break

            prev_eigenvalues = eigvals_small.clone()

        return eigvals_small, V

    def compute_single(self, A: torch.Tensor, num_iters: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute only the dominant (top-1) eigenpair via standard power iteration.

        Simpler and faster than block version when only the largest eigenvalue
        is needed (e.g., for spectral gap estimation).

        Args:
            A: Symmetric PSD matrix of shape (d, d).
            num_iters: Override iteration count.

        Returns:
            Tuple (eigenvalue, eigenvector) where eigenvalue is scalar and
            eigenvector is shape (d,).
        """
        num_iters = num_iters or self.num_iters
        A = A.to(self.device).float()

        v = torch.randn(self.d, dtype=torch.float32, device=self.device)
        v = v / v.norm()

        eigenvalue = torch.tensor(0.0, device=self.device)

        for _ in range(num_iters):
            w = A @ v
            eigenvalue = v @ w
            v = w / (w.norm() + 1e-12)

        return eigenvalue, v
