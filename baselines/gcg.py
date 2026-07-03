"""
GCG (Greedy Coordinate Gradient) baseline implementation.

This module implements the standard GCG attack for adversarial suffix
optimization, as described in:
"Universal and Transferable Adversarial Attacks on Aligned Language Models"
by Zou et al.

GCG performs greedy coordinate descent by:
  1. Computing the gradient of the loss w.r.t. the current suffix embeddings.
  2. Identifying the top-k token candidates for each position that maximize
     the negative gradient alignment.
  3. Evaluating a batch of candidate suffixes (sampling one coordinate per
     candidate).
  4. Selecting the candidate with the lowest loss and updating the suffix.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


class GCGAttack:
    """
    Greedy Coordinate Gradient (GCG) Attack.

    A baseline method for optimizing adversarial suffixes via greedy coordinate
descent with top-k candidate evaluation.

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
        Initialize the GCG attack.

        Args:
            model: The victim language model.
            tokenizer: The corresponding tokenizer.
            config: Optional dictionary of hyperparameters. Defaults are used
                for any missing keys.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Default hyperparameters matching the GCG paper
        default_config = {
            "batch_size": 512,
            "topk": 256,
            "num_steps": 500,
            "suffix_length": 20,
        }
        self.config = {**default_config, **(config or {})}

        # Embedding matrix
        self.embed_layer = self._get_input_embeddings()
        self.vocab_size = self.embed_layer.weight.shape[0]
        self.embed_dim = self.embed_layer.weight.shape[1]

    def _get_input_embeddings(self) -> nn.Embedding:
        """Retrieve the input embedding layer from the model."""
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        else:
            raise ValueError("Could not locate input embeddings in the model.")

    def compute_loss(
        self,
        suffix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the adversarial loss for a given suffix.

        Args:
            suffix_ids: Suffix token ids of shape (suffix_length,) or
                (batch_size, suffix_length).
            target_ids: Target token ids of shape (target_length,).
            prompt_embeds: Optional precomputed prompt embeddings.

        Returns:
            Scalar loss tensor (averaged over batch if applicable).
        """
        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        batch_size = suffix_ids.shape[0]
        suffix_ids = suffix_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Get suffix embeddings
        suffix_embeds = self.embed_layer(suffix_ids)  # (B, suffix_len, d)

        # Build full input embeddings
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
            prompt_embeds_batch = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            full_embeds = torch.cat([prompt_embeds_batch, suffix_embeds], dim=1)
        else:
            full_embeds = suffix_embeds

        # Forward pass
        outputs = self.model(inputs_embeds=full_embeds, output_hidden_states=False)
        logits = outputs.logits  # (B, seq_len, vocab_size)

        # Target logits: predict target tokens after the suffix
        suffix_len = suffix_embeds.shape[1]
        target_len = target_ids.shape[0]
        # logits for positions where target should be predicted
        target_logits = logits[:, suffix_len - 1 : suffix_len - 1 + target_len, :]

        # Repeat target_ids for batch
        target_ids_batch = target_ids.unsqueeze(0).expand(batch_size, -1)

        # Cross-entropy per sample
        loss = F.cross_entropy(
            target_logits.reshape(-1, self.vocab_size),
            target_ids_batch.reshape(-1),
            reduction="none",
        )
        loss = loss.view(batch_size, target_len).mean(dim=1)
        return loss

    def attack(
        self,
        prompt: str,
        target: str,
        behavior: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the GCG attack loop.

        Performs greedy coordinate descent with top-k candidate evaluation
        to find an adversarial suffix.

        Args:
            prompt: The user prompt (potentially harmful query).
            target: The desired target response string.
            behavior: Optional behavior descriptor (unused in baseline GCG,
                but kept for API compatibility).

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

        # Initialize suffix: generic affirmative or random tokens
        init_suffix_str = "Sure, here is" if behavior is None else f"Sure, here is {behavior}:"
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
            suffix_ids = torch.cat([init_suffix_ids, padding], dim=0)
        else:
            suffix_ids = init_suffix_ids[:suffix_length]

        # Tracking
        best_loss = float("inf")
        best_suffix_ids = suffix_ids.clone()
        losses: List[float] = []
        suffix_strings: List[str] = []

        batch_size = self.config["batch_size"]
        topk = self.config["topk"]

        for step in range(self.config["num_steps"]):
            # Compute gradient of loss w.r.t. suffix embeddings
            suffix_embeds = self.embed_layer(suffix_ids).detach().clone()
            suffix_embeds.requires_grad = True

            loss = self.compute_loss(suffix_ids, target_ids, prompt_embeds)
            current_loss = loss.item()
            losses.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix_ids = suffix_ids.clone()

            # Backward for gradient
            self.model.zero_grad()
            if suffix_embeds.grad is not None:
                suffix_embeds.grad.zero_()
            loss.mean().backward()

            grad = suffix_embeds.grad
            if grad is None:
                grad = torch.zeros_like(suffix_embeds)

            # Identify top-k candidate tokens for each position
            # grad shape: (suffix_length, embed_dim)
            # We want tokens that maximize -grad^T @ embed_weight^T
            # i.e., minimize grad^T @ embed_weight^T
            # candidate_scores shape: (suffix_length, vocab_size)
            with torch.no_grad():
                candidate_scores = torch.matmul(
                    grad, self.embed_layer.weight.T.to(self.device)
                )  # (suffix_length, vocab_size)

                # Top-k candidates per position (most negative = best descent)
                _, topk_indices = torch.topk(
                    -candidate_scores, k=topk, dim=-1
                )  # (suffix_length, topk)

                # Build candidate suffixes by sampling one position per candidate
                candidates = []
                for _ in range(batch_size):
                    # Randomly choose a position to modify
                    pos = torch.randint(0, suffix_length, (1,)).item()
                    # Randomly choose one of the top-k tokens for that position
                    token_idx = torch.randint(0, topk, (1,)).item()
                    new_token = topk_indices[pos, token_idx]

                    candidate = suffix_ids.clone()
                    candidate[pos] = new_token
                    candidates.append(candidate)

                candidates = torch.stack(candidates, dim=0)  # (batch_size, suffix_length)

                # Evaluate all candidates
                candidate_losses = self.compute_loss(candidates, target_ids, prompt_embeds)

                # Select best candidate
                best_idx = torch.argmin(candidate_losses)
                suffix_ids = candidates[best_idx].clone()

                # Decode for logging
                suffix_str = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                suffix_strings.append(suffix_str)

            # Periodic logging
            if (step + 1) % 50 == 0 or step == 0:
                print(
                    f"[GCG] Step {step + 1}/{self.config['num_steps']} | "
                    f"Loss: {current_loss:.4f} | Best: {best_loss:.4f} | "
                    f"Suffix: {suffix_str[:50]}"
                )

        # Final evaluation
        best_suffix_list = best_suffix_ids.cpu().tolist()
        best_suffix_str = self.tokenizer.decode(best_suffix_list, skip_special_tokens=True)

        return {
            "best_suffix": best_suffix_list,
            "best_loss": best_loss,
            "losses": losses,
            "suffix_strings": suffix_strings,
            "best_suffix_string": best_suffix_str,
        }
