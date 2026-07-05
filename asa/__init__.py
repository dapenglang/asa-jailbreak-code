"""
ASA (Active Subspace Attack) Jailbreak Framework
=================================================
Core algorithm modules for subspace-constrained adversarial optimization
on language model embeddings.

Modules:
    gumbel_softmax     -- Differentiable discrete sampling via Gumbel-Softmax
    afim               -- Streaming Fisher Information Matrix for subspace discovery
    incremental_svd    -- Efficient rank-k SVD updates via Brand's algorithm
    subspace_optimizer -- Gradient projection and coordinate descent in active subspace
    power_iteration    -- Block power iteration for efficient top-k eigendecomposition
    randomized_svd     -- Randomized SVD for large-scale matrix approximation (Halko et al.)
    hessian_subspace   -- Hessian-based active subspace via Lanczos + HVP (Pearlmutter)
    asa_attack         -- Main ASA attack loop (Algorithm 1)

Example:
    >>> from asa import AttackFisherInformationMatrix, IncrementalSVD, SubspaceOptimizer
    >>> from asa import BlockPowerIteration, RandomizedSVD, HessianSubspace
    >>> afim = AttackFisherInformationMatrix(d=4096, device='cuda')
    >>> svd = IncrementalSVD(d=4096, k=64, device='cuda')
    >>> opt = SubspaceOptimizer(U_k=svd.get_subspace()[0], device='cuda')
    >>> bpi = BlockPowerIteration(d=4096, k=32, device='cuda')
    >>> rsvd = RandomizedSVD(d=4096, k=32, device='cuda')
"""

from .gumbel_softmax import (
    gumbel_softmax,
    straight_through_gumbel_softmax,
    gumbel_max_sample,
    sample_gumbel,
)

from .afim import AttackFisherInformationMatrix
from .incremental_svd import IncrementalSVD
from .subspace_optimizer import SubspaceOptimizer
from .power_iteration import BlockPowerIteration
from .randomized_svd import RandomizedSVD
from .hessian_subspace import HessianSubspace

__all__ = [
    # gumbel_softmax
    "gumbel_softmax",
    "straight_through_gumbel_softmax",
    "gumbel_max_sample",
    "sample_gumbel",
    # core classes
    "AttackFisherInformationMatrix",
    "IncrementalSVD",
    "SubspaceOptimizer",
    # spectral analysis methods
    "BlockPowerIteration",
    "RandomizedSVD",
    "HessianSubspace",
]

__version__ = "0.2.0"
