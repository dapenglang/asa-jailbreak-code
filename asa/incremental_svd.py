"""
Incremental Rank-k SVD via Brand's Algorithm
=============================================
Efficiently maintains a truncated SVD of a data matrix as new
observations (gradient vectors) arrive in a streaming fashion.

Complexity per update: O(d * k^2) instead of O(d^3) for full SVD.

Reference:
    Brand, M. (2006). Fast low-rank modifications of the thin
    singular value decomposition. Linear Algebra and its Applications.
"""

import math
from typing import Optional, Tuple

import torch


class IncrementalSVD:
    """
    Incremental rank-k SVD updater.

    Maintains an approximate truncated SVD  X  ≈  U_k Sigma_k V_k^T
    where X is the (implicitly stored) data matrix whose columns are
    the observed gradient vectors. When a new vector x arrives, the
    SVD is updated in O(d k^2) time without recomputing from scratch.

    Attributes:
        d (int): Dimension of observation vectors.
        k (int): Rank of the truncated SVD to maintain.
        device (torch.device): Compute device.
    """

    def __init__(self, d: int, k: int, device: str = "cuda") -> None:
        """
        Initialize the incremental SVD state.

        Args:
            d: Dimensionality of incoming vectors.
            k: Target rank (must satisfy 1 <= k <= d).
            device: PyTorch device string.
        """
        if not (1 <= k <= d):
            raise ValueError(f"Rank k must satisfy 1 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # U: (d, k)  orthonormal columns
        # S: (k,)    singular values
        # V: (t, k)  right singular vectors (t = number of observations seen)
        # We keep V implicitly small by only tracking the rotated coordinates.
        self.U = torch.zeros((d, k), dtype=torch.float32, device=self.device)
        self.S = torch.zeros(k, dtype=torch.float32, device=self.device)
        self.V = None  # Will be created lazily or kept small
        self.t = 0     # Number of observations processed

    def _initialize(self, x: torch.Tensor) -> None:
        """
        Bootstrap the SVD with the very first observation.
        """
        norm = x.norm()
        if norm > 1e-12:
            self.U[:, 0] = x / norm
            self.S[0] = norm
        else:
            # Fallback: random orthonormal vector
            u = torch.randn(self.d, device=self.device)
            u = u / u.norm()
            self.U[:, 0] = u
            self.S[0] = 0.0

        # V starts as a 1 x k matrix: [1, 0, 0, ...]
        self.V = torch.zeros((1, self.k), dtype=torch.float32, device=self.device)
        self.V[0, 0] = 1.0
        self.t = 1

    def update(self, x_t: torch.Tensor) -> None:
        """
        Update the rank-k SVD with a new observation vector x_t.

        Implements a practical variant of Brand's incremental SVD:
            1. Compute the residual of x_t orthogonal to current U.
            2. Form an augmented SVD problem of size (k+1) or (k+2).
            3. Truncate back to rank k.

        Args:
            x_t: New observation vector of shape (d,) or (d, 1).
        """
        if x_t.dim() == 2:
            x_t = x_t.squeeze(1)
        if x_t.shape != (self.d,):
            raise ValueError(f"Expected x_t shape ({self.d},), got {x_t.shape}")

        # Bootstrap on first call
        if self.t == 0:
            self._initialize(x_t)
            return

        # ------------------------------------------------------------------
        # Step 1: projection onto current subspace and orthogonal residual
        # ------------------------------------------------------------------
        # m = U^T x   (k,)
        m = self.U.t() @ x_t

        # p = x - U m  (residual component)
        p = x_t - self.U @ m
        r_norm = p.norm()

        # ------------------------------------------------------------------
        # Step 2: form the small update matrix and its SVD
        # ------------------------------------------------------------------
        if r_norm > 1e-12:
            # Augmented problem is (k+2) dimensional
            r = p / r_norm

            # Build the small matrix K = [ diag(S)  m    0
            #                              0        r_norm 0 ]
            # Actually for a single new column x_t appended to X,
            # the updated SVD can be computed from:
            #   [ U, r ]  *  K  *  [ V^T, 0; 0, 1 ]
            # where K is a (k+1) x (k+1) matrix:
            #   K = [ diag(S)   m
            #         0^T       r_norm ]
            # But we also need to account for the new row in V.
            # Simpler practical approach: build the (k+1)x(k+1) matrix.

            k1 = self.k + 1
            K = torch.zeros((k1, k1), dtype=torch.float32, device=self.device)
            K[: self.k, : self.k] = torch.diag(self.S)
            K[: self.k, self.k] = m
            K[self.k, : self.k] = 0.0
            K[self.k, self.k] = r_norm

            # SVD of K
            U_k, S_k, Vh_k = torch.linalg.svd(K, full_matrices=False)

            # Truncate to rank k
            self.S = S_k[: self.k]

            # Update U: [U, r] @ U_k[:, :k]
            U_r = torch.cat([self.U, r.unsqueeze(1)], dim=1)  # (d, k+1)
            self.U = U_r @ U_k[:, : self.k]  # (d, k)

            # Update V implicitly: extend by one row, then rotate
            if self.V is not None:
                V_ext = torch.cat(
                    [self.V, torch.zeros((1, self.k), device=self.device)], dim=0
                )  # (t+1, k)
                # New row for the new observation is [0,...,0,1] in the augmented basis
                # After rotation by V_k, the new V is:
                V_rot = torch.zeros(
                    (self.t + 1, k1), dtype=torch.float32, device=self.device
                )
                V_rot[: self.t, : self.k] = self.V
                V_rot[self.t, self.k] = 1.0
                self.V = V_rot @ Vh_k.t()[:, : self.k]
            else:
                self.V = Vh_k.t()[: self.t + 1, : self.k]
        else:
            # Residual is negligible: rank does not increase
            # K = [ diag(S) ; m^T ]  (k x k) after appending x_t as new column
            # Actually x lies in span(U), so the update matrix is just
            #   K = diag(S) + something; easiest is:
            K = torch.diag(self.S)
            # But we need the column update. For a new column x = U m:
            # The augmented representation is K = [diag(S), m] with a new row in V.
            # Simpler: build (k+1) x (k+1) matrix with zero residual.
            k1 = self.k + 1
            K = torch.zeros((k1, k1), dtype=torch.float32, device=self.device)
            K[: self.k, : self.k] = torch.diag(self.S)
            K[: self.k, self.k] = m
            K[self.k, : self.k] = 0.0
            K[self.k, self.k] = 0.0

            U_k, S_k, Vh_k = torch.linalg.svd(K, full_matrices=False)
            self.S = S_k[: self.k]
            self.U = self.U @ U_k[: self.k, : self.k]

            if self.V is not None:
                V_rot = torch.zeros(
                    (self.t + 1, k1), dtype=torch.float32, device=self.device
                )
                V_rot[: self.t, : self.k] = self.V
                V_rot[self.t, self.k] = 1.0
                self.V = V_rot @ Vh_k.t()[:, : self.k]
            else:
                self.V = Vh_k.t()[: self.t + 1, : self.k]

        self.t += 1

    def get_subspace(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Return the current rank-k truncated SVD factors.

        Returns:
            A tuple ``(U_k, Sigma_k, V_k)`` where:
                - U_k:     (d, k) left singular vectors.
                - Sigma_k: (k,)   singular values.
                - V_k:     (t, k) right singular vectors, or None if
                           no observations have been processed yet.
        """
        return self.U, self.S, self.V

    def get_complexity(self) -> int:
        """
        Return the time complexity of a single rank-1 update.

        Brand's incremental SVD requires O(d * k^2) operations per
        update: O(dk) for projection, O(k^3) for the small SVD,
        and O(dk^2) for the basis rotation.

        Returns:
            Integer proportional to d * k^2.
        """
        return self.d * self.k * self.k
