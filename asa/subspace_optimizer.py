"""
Subspace Optimizer for Active Subspace Attack
===============================================
Projects full-dimensional gradients into a low-rank active subspace
and performs efficient coordinate descent to select discrete tokens.

Key operations:
    1. Gradient projection:   g_tilde = U_k U_k^T g_full
    2. Subspace token search: find the token whose embedding best
                              aligns with the projected gradient.
    3. SNR monitoring:        measure how much signal is preserved
                              by the subspace projection.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class SubspaceOptimizer:
    """
    Optimizer that restricts gradient updates to an active subspace.

    Given an orthonormal basis U_k for the active subspace (from AFIM
    or IncrementalSVD), this class:
        - Projects gradients into the subspace.
        - Searches for the best discrete token in the projected direction.
        - Tracks the signal-to-noise ratio of the projection.

    Attributes:
        U_k (torch.Tensor): (d, k) orthonormal basis matrix.
        k (int): Subspace dimension.
        device (torch.device): Compute device.
    """

    def __init__(self, U_k: torch.Tensor, device: str = "cuda") -> None:
        """
        Initialize the subspace optimizer.

        Args:
            U_k: (d, k) tensor whose columns form an orthonormal basis
                 for the active subspace. Typically obtained from the
                 top-k eigenvectors of the AFIM or left singular vectors
                 of IncrementalSVD.
            device: PyTorch device string.
        """
        if U_k.dim() != 2:
            raise ValueError(f"U_k must be 2D, got shape {U_k.shape}")

        self.d, self.k = U_k.shape
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Ensure U_k is on the correct device and orthonormal
        self.U_k = U_k.to(self.device)
        self._validate_orthonormality()

    def _validate_orthonormality(self, tol: float = 1e-3) -> None:
        """
        Sanity check: U_k^T U_k should be close to identity.
        """
        gram = self.U_k.t() @ self.U_k
        eye = torch.eye(self.k, device=self.device)
        deviation = (gram - eye).abs().max().item()
        if deviation > tol:
            # Re-orthogonalize via QR if deviation is large
            q, _ = torch.linalg.qr(self.U_k)
            self.U_k = q[:, : self.k]

    def project_gradient(self, g_full: torch.Tensor) -> torch.Tensor:
        """
        Project a full-dimensional gradient into the active subspace.

        The projection operator is  P = U_k U_k^T.
        The projected gradient lives in the column space of U_k but
        is represented in the original d-dimensional embedding space.

        Args:
            g_full: Gradient vector of shape (d,) or (d, 1).

        Returns:
            Projected gradient g_tilde of shape (d,).
        """
        if g_full.dim() == 2:
            g_full = g_full.squeeze(1)
        if g_full.shape != (self.d,):
            raise ValueError(f"Expected gradient shape ({self.d},), got {g_full.shape}")

        # g_tilde = U_k (U_k^T g_full)
        coeffs = self.U_k.t() @ g_full  # (k,)
        g_tilde = self.U_k @ coeffs     # (d,)
        return g_tilde

    def subspace_coordinate_descent(
        self,
        g_tilde: torch.Tensor,
        embedding_matrix: torch.Tensor,
        top_k_candidates: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the token whose embedding best aligns with the projected
        gradient direction, restricted to a candidate set for efficiency.

        This implements a greedy coordinate step in the subspace:
            i* = argmax_i  <e_i, g_tilde>
        where e_i is the i-th row of the embedding matrix.

        In practice we first reduce the search to ``top_k_candidates``
        via fast approximate scoring, then do exact argmax on that subset.

        Args:
            g_tilde: Projected gradient of shape (d,) or (d, 1).
            embedding_matrix: (V, d) token embedding matrix where V is
                              the vocabulary size.
            top_k_candidates: Number of candidates to score exactly.
                              Larger -> more exact but slower.

        Returns:
            A tuple ``(best_token_ids, best_scores)`` where:
                - best_token_ids: Long tensor of shape (num_positions,)
                  containing the selected token indices.
                - best_scores:    Float tensor of shape (num_positions,)
                  containing the alignment scores.
        """
        if g_tilde.dim() == 2:
            g_tilde = g_tilde.squeeze(1)
        if g_tilde.shape != (self.d,):
            raise ValueError(
                f"Expected g_tilde shape ({self.d},), got {g_tilde.shape}"
            )

        vocab_size, embed_dim = embedding_matrix.shape
        if embed_dim != self.d:
            raise ValueError(
                f"Embedding dim {embed_dim} does not match subspace dim {self.d}"
            )

        # Move embedding matrix to device if needed
        if embedding_matrix.device != self.device:
            embedding_matrix = embedding_matrix.to(self.device)

        # If vocabulary is small enough, just do exact search
        if vocab_size <= top_k_candidates:
            scores = embedding_matrix @ g_tilde  # (V,)
            best_idx = scores.argmax()
            return best_idx.unsqueeze(0), scores[best_idx].unsqueeze(0)

        # ------------------------------------------------------------------
        # Fast approximate pre-filtering via random projection or subspace
        # scoring. Here we use the subspace basis directly: project both
        # embeddings and gradient to the k-dimensional subspace, then score.
        # Complexity: O(V * k) instead of O(V * d).
        # ------------------------------------------------------------------
        # E_proj = E @ U_k  -> (V, k)
        E_proj = embedding_matrix @ self.U_k
        g_proj = self.U_k.t() @ g_tilde  # (k,)

        # Approximate scores in k-dim space
        approx_scores = E_proj @ g_proj  # (V,)

        # Select top-k candidates
        _, candidate_indices = torch.topk(approx_scores, k=top_k_candidates, largest=True)

        # Exact scoring on candidates in full d-dimensional space
        candidate_embeddings = embedding_matrix[candidate_indices]  # (top_k, d)
        exact_scores = candidate_embeddings @ g_tilde  # (top_k,)

        best_local_idx = exact_scores.argmax()
        best_idx = candidate_indices[best_local_idx]
        best_score = exact_scores[best_local_idx]

        return best_idx.unsqueeze(0), best_score.unsqueeze(0)

    def compute_snr(self, g_full: torch.Tensor, g_tilde: torch.Tensor) -> torch.Tensor:
        """
        Compute the signal-to-noise ratio of the subspace projection.

        SNR = ||g_tilde||^2 / ||g_full - g_tilde||^2

        A high SNR means the active subspace captures most of the
        gradient energy, justifying the low-rank approximation.

        Args:
            g_full: Original full gradient of shape (d,) or (d, 1).
            g_tilde: Projected gradient of shape (d,) or (d, 1).

        Returns:
            Scalar SNR tensor. Returns +inf if the residual norm is zero.
        """
        if g_full.dim() == 2:
            g_full = g_full.squeeze(1)
        if g_tilde.dim() == 2:
            g_tilde = g_tilde.squeeze(1)

        signal_norm_sq = g_tilde.norm() ** 2
        residual = g_full - g_tilde
        noise_norm_sq = residual.norm() ** 2

        if noise_norm_sq.item() < 1e-18:
            return torch.tensor(float("inf"), device=self.device)

        return signal_norm_sq / noise_norm_sq
