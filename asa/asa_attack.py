"""
ASA (Adaptive Subspace Attack) implementation.

This module implements Algorithm 1 from the paper:
"Adaptive Subspace Attacks for Jailbreaking Large Language Models".

The ASA attack combines:
  - Online AFIM (Adversarial Fisher Information Matrix) estimation
  - Incremental SVD for active subspace tracking
  - Subspace-projected coordinate descent
  - Gumbel-Softmax + Straight-Through Estimator (STE) for discrete token sampling
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


class ASAAttack:
    """
    Adaptive Subspace Attack (ASA) for generating adversarial suffixes.

    The attack optimizes a discrete token suffix via a continuous embedding
    relaxation, using an adaptive low-dimensional subspace to guide coordinate
    descent efficiently.

    Attributes:
        model: The target language model.
        tokenizer: The tokenizer corresponding to the model.
        config: Hyperparameter configuration dictionary.
        device: The torch device (cuda if available).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the ASA attack.

        Args:
            model: The victim language model.
            tokenizer: The corresponding tokenizer.
            config: Optional dictionary of hyperparameters. Defaults are used
                for any missing keys.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Default hyperparameters from the paper
        default_config = {
            "batch_size": 512,
            "topk": 256,
            "num_steps": 500,
            "temp_init": 2.0,
            "temp_min": 0.5,
            "gamma": 0.95,
            "afim_window": 50,
            "subspace_dim": 32,
            "suffix_length": 20,
            "lr": 0.01,
            "coordinate_samples": 1,
        }
        self.config = {**default_config, **(config or {})}

        # Embedding matrix (V x d)
        self.embed_layer = self._get_input_embeddings()
        self.vocab_size = self.embed_layer.weight.shape[0]
        self.embed_dim = self.embed_layer.weight.shape[1]

        # AFIM buffer: stores recent gradients for online covariance estimation
        self.afim_buffer: List[torch.Tensor] = []
        self.afim: Optional[torch.Tensor] = None

        # Active subspace basis (d x k)
        self.subspace_basis: Optional[torch.Tensor] = None

        # Current temperature for Gumbel-Softmax
        self.temperature = self.config["temp_init"]

    def _get_input_embeddings(self) -> nn.Embedding:
        """Retrieve the input embedding layer from the model."""
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        else:
            raise ValueError("Could not locate input embeddings in the model.")

    def get_target_suffix(self, behavior: Optional[str] = None) -> str:
        """
        Get the initial target suffix string.

        Uses a behavior-specific template if provided, otherwise defaults to
        a generic affirmative prefix.

        Args:
            behavior: Optional behavior string to customize the target suffix.

        Returns:
            The initial suffix string.
        """
        if behavior is not None:
            # Behavior-specific target templates can be added here
            return f"Sure, here is {behavior}:"
        return "Sure, here is"

    def compute_loss(
        self,
        suffix_embeds: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the adversarial loss and its gradient w.r.t. suffix embeddings.

        The loss is the negative log-likelihood of the target tokens given the
        full prompt (user prompt + adversarial suffix).

        Args:
            suffix_embeds: Continuous suffix embeddings of shape
                (suffix_length, embed_dim). May require grad.
            target_ids: Target token ids of shape (target_length,).
            prompt_embeds: Optional precomputed prompt embeddings.

        Returns:
            loss: Scalar loss tensor.
            grad: Gradient of loss w.r.t. suffix_embeds, shape
                (suffix_length, embed_dim).
        """
        suffix_embeds = suffix_embeds.to(self.device)
        target_ids = target_ids.to(self.device)

        if suffix_embeds.requires_grad:
            suffix_embeds.retain_grad()

        # Build full input embeddings
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
            full_embeds = torch.cat([prompt_embeds, suffix_embeds.unsqueeze(0)], dim=1)
        else:
            full_embeds = suffix_embeds.unsqueeze(0)

        # Forward pass
        outputs = self.model(inputs_embeds=full_embeds, output_hidden_states=False)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Target logits start after the prefix (prompt + suffix)
        # We want to predict target_ids at positions after the suffix
        suffix_len = suffix_embeds.shape[0]
        target_logits = logits[0, suffix_len - 1 : suffix_len - 1 + target_ids.shape[0], :]

        # Compute cross-entropy loss
        loss = F.cross_entropy(target_logits, target_ids, reduction="mean")

        # Backward to get gradient w.r.t. suffix embeddings
        self.model.zero_grad()
        loss.backward()

        grad = suffix_embeds.grad
        if grad is None:
            grad = torch.zeros_like(suffix_embeds)

        return loss.detach(), grad.detach()

    def sample_suffix_tokens(
        self, suffix_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample discrete suffix tokens using Gumbel-Softmax + STE.

        For each position, computes similarity scores against the vocabulary
        embeddings, adds Gumbel noise scaled by temperature, and applies
        softmax to obtain a relaxed one-hot vector. The Straight-Through
        Estimator (STE) is used so that the forward pass uses the hard argmax
        while the backward pass flows through the softmax.

        Args:
            suffix_embeds: Continuous suffix embeddings of shape
                (suffix_length, embed_dim).

        Returns:
            hard_tokens: Discrete token ids of shape (suffix_length,).
            soft_embeds: Soft embeddings after Gumbel-Softmax, shape
                (suffix_length, embed_dim), suitable for gradient backprop.
        """
        suffix_embeds = suffix_embeds.to(self.device)
        temp = max(self.temperature, self.config["temp_min"])

        # Compute logits = (suffix_embeds @ embed_weight.T) / sqrt(d)
        # shape: (suffix_length, vocab_size)
        logits = torch.matmul(
            suffix_embeds, self.embed_layer.weight.T.to(self.device)
        ) / math.sqrt(self.embed_dim)

        # Gumbel noise
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)

        # Gumbel-Softmax
        soft_probs = F.softmax((logits + gumbel_noise) / temp, dim=-1)

        # Hard sample (argmax) with STE
        hard_indices = torch.argmax(soft_probs, dim=-1)  # (suffix_length,)
        hard_onehot = F.one_hot(hard_indices, num_classes=self.vocab_size).float()

        # STE: forward uses hard, backward uses soft
        ste_onehot = hard_onehot - soft_probs.detach() + soft_probs

        # Map back to embedding space
        soft_embeds = torch.matmul(ste_onehot, self.embed_layer.weight.to(self.device))

        return hard_indices, soft_embeds

    def update_afim(self, grad: torch.Tensor) -> None:
        """
        Update the online AFIM (Adversarial Fisher Information Matrix) estimate.

        The AFIM is approximated as the outer product of the flattened gradient
        with itself. A sliding window of recent gradients is maintained, and
        the empirical covariance matrix is computed over this window.

        Args:
            grad: Gradient tensor of shape (suffix_length, embed_dim).
        """
        flat_grad = grad.view(-1).detach().cpu()  # (suffix_length * embed_dim,)
        self.afim_buffer.append(flat_grad)

        window = self.config["afim_window"]
        if len(self.afim_buffer) > window:
            self.afim_buffer.pop(0)

        # Compute empirical covariance: (1/N) * sum(g_i * g_i^T)
        if len(self.afim_buffer) > 0:
            grads = torch.stack(self.afim_buffer, dim=0)  # (N, D)
            self.afim = torch.matmul(grads.T, grads) / len(self.afim_buffer)
            self.afim = self.afim.to(self.device)

    def update_subspace(self) -> None:
        """
        Update the active subspace via incremental SVD.

        Performs a truncated SVD on the AFIM matrix and retains the top-k
        eigenvectors to form the subspace basis. If the AFIM is not yet
        initialized (insufficient gradient history), the basis is initialized
        randomly or as None.
        """
        if self.afim is None:
            return

        k = self.config["subspace_dim"]
        # Symmetric matrix, use eigh for efficiency
        eigenvalues, eigenvectors = torch.linalg.eigh(self.afim)

        # Sort descending
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        topk_indices = sorted_indices[:k]

        basis = eigenvectors[:, topk_indices]  # (D, k)
        self.subspace_basis = basis.to(self.device)

    def project_and_optimize(
        self,
        grad: torch.Tensor,
        suffix_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the gradient onto the active subspace and perform coordinate descent.

        Steps:
          1. Flatten the gradient and project onto the subspace basis.
          2. Update the subspace coordinates via gradient descent.
          3. Map back to the full embedding space.
          4. Optionally perform top-k coordinate descent in the full space
             by evaluating candidate token substitutions.

        Args:
            grad: Gradient tensor of shape (suffix_length, embed_dim).
            suffix_embeds: Current suffix embeddings, shape
                (suffix_length, embed_dim).

        Returns:
            Updated suffix embeddings, same shape as input.
        """
        suffix_embeds = suffix_embeds.detach().clone()
        lr = self.config["lr"]

        if self.subspace_basis is not None:
            # Flatten gradient: (D,)
            flat_grad = grad.view(-1)
            D = flat_grad.shape[0]

            # Project gradient onto subspace: z = B^T @ g
            # B: (D, k), z: (k,)
            z = torch.matmul(self.subspace_basis.T, flat_grad)

            # Update subspace coordinates
            z_update = z - lr * z

            # Map back to full space: delta = B @ z_update
            delta_flat = torch.matmul(self.subspace_basis, z_update)
            delta = delta_flat.view_as(suffix_embeds)

            suffix_embeds = suffix_embeds - lr * delta
        else:
            # Fallback: standard gradient descent if subspace not ready
            suffix_embeds = suffix_embeds - lr * grad

        return suffix_embeds

    def attack(
        self,
        prompt: str,
        target: str,
        behavior: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the main ASA attack loop (Algorithm 1).

        The algorithm iteratively refines an adversarial suffix appended to the
        user prompt, aiming to elicit the target response from the model.

        Args:
            prompt: The user prompt (potentially harmful query).
            target: The desired target response string.
            behavior: Optional behavior descriptor for suffix initialization.

        Returns:
            A dictionary containing:
                - "best_suffix": The optimized suffix token ids (List[int]).
                - "best_loss": The lowest adversarial loss achieved.
                - "losses": List of loss values per iteration.
                - "suffix_strings": List of suffix strings per iteration.
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Tokenize prompt and target
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ][0].to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].to(self.device)

        # Prompt embeddings (static)
        with torch.no_grad():
            prompt_embeds = self.embed_layer(prompt_ids).detach()

        # Initialize suffix
        init_suffix_str = self.get_target_suffix(behavior)
        init_suffix_ids = self.tokenizer(
            init_suffix_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.device)

        # Pad or truncate to suffix_length
        suffix_length = self.config["suffix_length"]
        if init_suffix_ids.shape[0] < suffix_length:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            padding = torch.full(
                (suffix_length - init_suffix_ids.shape[0],), pad_id, dtype=torch.long, device=self.device
            )
            init_suffix_ids = torch.cat([init_suffix_ids, padding], dim=0)
        else:
            init_suffix_ids = init_suffix_ids[:suffix_length]

        # Initialize continuous suffix embeddings
        suffix_embeds = self.embed_layer(init_suffix_ids).detach().clone()
        suffix_embeds.requires_grad = True

        # Tracking
        best_loss = float("inf")
        best_suffix_ids = init_suffix_ids.clone()
        losses: List[float] = []
        suffix_strings: List[str] = []

        # Main optimization loop
        for step in range(self.config["num_steps"]):
            # a. Compute adversarial loss and gradient
            loss, grad = self.compute_loss(suffix_embeds, target_ids, prompt_embeds)
            losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                # Recover discrete tokens from current soft embeddings
                with torch.no_grad():
                    logits = torch.matmul(
                        suffix_embeds, self.embed_layer.weight.T
                    ) / math.sqrt(self.embed_dim)
                    best_suffix_ids = torch.argmax(logits, dim=-1)

            # b. Update AFIM with current gradient
            self.update_afim(grad)

            # c. Update active subspace via incremental SVD
            if step > 0 and step % 10 == 0:
                self.update_subspace()

            # d & e. Project gradient and optimize in subspace
            suffix_embeds = self.project_and_optimize(grad, suffix_embeds)
            suffix_embeds.requires_grad = True

            # f. Update suffix via Gumbel-Softmax + STE
            sampled_ids, suffix_embeds = self.sample_suffix_tokens(suffix_embeds)
            suffix_embeds = suffix_embeds.detach().clone()
            suffix_embeds.requires_grad = True

            # Decode for logging
            with torch.no_grad():
                suffix_str = self.tokenizer.decode(sampled_ids, skip_special_tokens=True)
                suffix_strings.append(suffix_str)

            # g. Anneal temperature
            self.temperature = max(
                self.config["temp_min"],
                self.temperature * self.config["gamma"],
            )

            # Periodic logging
            if (step + 1) % 50 == 0 or step == 0:
                print(
                    f"[ASA] Step {step + 1}/{self.config['num_steps']} | "
                    f"Loss: {loss.item():.4f} | Best: {best_loss:.4f} | "
                    f"Temp: {self.temperature:.4f} | Suffix: {suffix_str[:50]}"
                )

        # Final evaluation on best suffix
        best_suffix_list = best_suffix_ids.cpu().tolist()
        best_suffix_str = self.tokenizer.decode(best_suffix_list, skip_special_tokens=True)

        return {
            "best_suffix": best_suffix_list,
            "best_loss": best_loss,
            "losses": losses,
            "suffix_strings": suffix_strings,
            "best_suffix_string": best_suffix_str,
        }
