"""
Gumbel-Softmax Utilities for Discrete Token Sampling
======================================================
Provides differentiable approximations to categorical sampling,
essential for end-to-end gradient-based optimization over discrete
token choices in adversarial attacks.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def sample_gumbel(shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Sample from the standard Gumbel(0, 1) distribution.

    The Gumbel distribution is used to convert a softmax probability
    into an argmax operation while keeping the sampling differentiable.
    We use the standard inverse-transform sampling:
        g = -log(-log(u))
    where u ~ Uniform(0, 1).

    To avoid numerical issues with log(0), we clamp u to (eps, 1-eps).

    Args:
        shape: Output tensor shape.
        device: PyTorch device (e.g., 'cuda' or 'cpu').

    Returns:
        Tensor of the given shape drawn from Gumbel(0, 1).
    """
    eps = 1e-20
    u = torch.rand(shape, device=device)
    u = u.clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u))


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """
    Gumbel-Softmax (a.k.a. Concrete) distribution sampler.

    Given logits, adds Gumbel noise and applies a temperature-scaled
    softmax to produce a soft (differentiable) one-hot vector.

    When ``hard=True``, returns a one-hot vector in the forward pass
    but backpropagates through the soft probabilities (STE-like).

    Math:
        y = softmax((logits + g) / tau)
        where g ~ Gumbel(0, 1)

    Args:
        logits: Unnormalized log probabilities, shape (*, num_classes).
        temperature: Softmax temperature. Lower -> closer to one-hot.
                     Typical range: (0.1, 2.0).
        hard: If True, forward pass returns argmax (discrete), but
              gradients flow through the soft samples.
        dim: Dimension along which to apply softmax.

    Returns:
        Soft or hard sample from the Gumbel-Softmax distribution.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Add Gumbel noise
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=dim)

    if hard:
        # Straight-through trick: forward uses argmax, backward uses y_soft
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        # y_hard - y_soft is zero in forward (so y_hard is returned),
        # but in backward the gradient flows through y_soft
        return y_hard - y_soft.detach() + y_soft

    return y_soft


def straight_through_gumbel_softmax(
    logits: torch.Tensor,
    temperature: float,
    dim: int = -1,
) -> torch.Tensor:
    """
    Straight-Through Gumbel-Softmax (STE variant).

    Forward:  returns a discrete one-hot vector (argmax).
    Backward: gradients flow through the continuous Gumbel-Softmax
              relaxation.

    This is a convenience wrapper around ``gumbel_softmax`` with
    ``hard=True`` but explicitly naming the STE behavior.

    Args:
        logits: Unnormalized log probabilities, shape (*, num_classes).
        temperature: Softmax temperature.
        dim: Dimension along which to apply softmax.

    Returns:
        Discrete one-hot sample in forward, soft gradients in backward.
    """
    return gumbel_softmax(logits, temperature=temperature, hard=True, dim=dim)


def gumbel_max_sample(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Non-differentiable Gumbel-Max sampling.

    Returns a hard one-hot sample without any continuous relaxation.
    Useful when you need a purely discrete sample and do not require
    gradients to flow through the sampling step.

    Args:
        logits: Unnormalized log probabilities.
        dim: Dimension along which to sample.

    Returns:
        One-hot tensor of the same shape as logits.
    """
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    index = (logits + gumbel_noise).argmax(dim=dim, keepdim=True)
    return torch.zeros_like(logits).scatter_(dim, index, 1.0)
