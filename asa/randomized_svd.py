"""
Randomized SVD for Large-Scale Matrix Approximation
=====================================================
Efficient low-rank matrix approximation via randomized projections.

Instead of computing the full O(d^3) eigendecomposition, randomized SVD
projects the matrix onto a random subspace of dimension (k + p), then
computes an economic SVD on the much smaller projected matrix.

Complexity: O(d^2 * (k + p)) instead of O(d^3).
Memory: O(d * (k + p)) for the sketch.

Reference:
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions.
    SIAM Review, 53(2), 217-288.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class RandomizedSVD:
    """
    Randomized SVD for approximate low-rank decomposition.

    Given a symmetric PSD matrix A (d x d), computes:
        A ≈ U_k @ diag(S_k) @ V_k^T
    where U_k (d x k), S_k (k,), V_k (k x k).

    The algorithm:
        1. Draw random test matrix Omega (d x l) where l = k + oversampling
        2. Form the range sketch Y = A @ Omega
        3. Orthonormalize Y via QR: Q = qr(Y)
        4. Form the small matrix B = Q^T @ A @ Q (l x l)
        5. Eigendecompose B: B = W @ diag(lambda) @ W^T
        6. Map back: U_k = Q @ W[:, :k], eigenvalues = lambda[:k]

    Attributes:
        d (int): Matrix dimension.
        k (int): Target rank.
        oversampling (int): Extra random projections for accuracy (default: 10).
        num_power_iters (int): Power iteration passes to improve approximation (default: 2).
        device (torch.device): Compute device.
    """

    def __init__(
        self,
        d: int,
        k: int,
        oversampling: int = 10,
        num_power_iters: int = 2,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the randomized SVD solver.

        Args:
            d: Dimension of the matrix.
            k: Target rank (k <= d).
            oversampling: Oversampling parameter p (default: 10).
            num_power_iters: Number of power iteration passes (default: 2).
            device: PyTorch device string.
        """
        if not (1 <= k <= d):
            raise ValueError(f"k must satisfy 1 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k
        self.oversampling = oversampling
        self.num_power_iters = num_power_iters
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute(
        self,
        A: torch.Tensor,
        k: Optional[int] = None,
        center: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute approximate top-k eigenpairs via randomized SVD.

        Args:
            A: Symmetric PSD matrix of shape (d, d).
            k: Override target rank (default: self.k).
            center: Whether to subtract the mean eigenvalue for numerical
                    stability (default: True).

        Returns:
            Tuple (eigenvalues, eigenvectors):
                - eigenvalues:  shape (k,), descending order.
                - eigenvectors: shape (d, k), orthonormal columns.
        """
        if A.shape != (self.d, self.d):
            raise ValueError(f"Expected A shape ({self.d}, {self.d}), got {A.shape}")

        k = k or self.k
        l = k + self.oversampling  # sketch dimension
        l = min(l, self.d)
        A = A.to(self.device).float()

        # Optional centering for better numerical behavior
        if center:
            trace_val = torch.trace(A) / self.d
            A = A - trace_val * torch.eye(self.d, device=self.device)

        # Step 1: Draw random test matrix
        Omega = torch.randn(self.d, l, dtype=torch.float32, device=self.device)
        Y = A @ Omega  # (d, l)

        # Step 2: Power iteration to improve approximation
        Q, _ = torch.linalg.qr(Y)
        for _ in range(self.num_power_iters):
            Z = A @ Q
            Q, _ = torch.linalg.qr(Z)

        # Step 3: Form small matrix B = Q^T A Q (l x l)
        B = Q.t() @ A @ Q  # (l, l)

        # Step 4: Eigendecompose the small matrix
        eigenvalues_small, eigenvectors_small = torch.linalg.eigh(B)

        # Sort descending
        idx = torch.argsort(eigenvalues_small, descending=True)
        eigenvalues_small = eigenvalues_small[idx][:k]
        eigenvectors_small = eigenvectors_small[:, idx][:, :k]

        # Step 5: Map back to original space
        eigenvectors = Q @ eigenvectors_small  # (d, k)

        # Restore centering offset if applied
        if center:
            eigenvalues_small = eigenvalues_small + trace_val

        return eigenvalues_small, eigenvectors

    def incremental_update(
        self,
        Q: torch.Tensor,
        A: torch.Tensor,
        x_new: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update an existing sketch Q with a new observation vector.

        Instead of recomputing from scratch, this updates the QR sketch
        incrementally by appending the new projected vector and re-orthogonalizing.

        Args:
            Q: Current orthonormal basis of shape (d, l).
            A: The matrix being approximated (d, d).
            x_new: New observation vector of shape (d,).

        Returns:
            Updated orthonormal basis Q of shape (d, l).
        """
        # Project new vector through A and append to Q
        y_new = A @ x_new.unsqueeze(1)  # (d, 1)
        Q_aug = torch.cat([Q, y_new], dim=1)  # (d, l+1)
        Q_new, _ = torch.linalg.qr(Q_aug)
        return Q_new[:, :-1]  # Keep only first l columns
