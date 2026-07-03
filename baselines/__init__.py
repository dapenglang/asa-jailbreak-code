"""
Baselines package for adversarial suffix attacks.

Exports:
    GCGAttack: The Greedy Coordinate Gradient baseline attack.
"""

from .gcg import GCGAttack

__all__ = ["GCGAttack"]
