"""
ASA Jailbreak Framework - Evaluation Module

提供攻击评估、防御评估以及各类评价指标。
"""

from .metrics import (
    compute_asr,
    compute_perplexity,
    compute_refusal_rate,
    compute_transfer_asr,
    judge_harmbench,
    judge_keyword,
)

from .defenses import (
    PerplexityFilter,
    SmoothLLM,
    SafeDecoding,
    DefenseEvaluator,
)

__all__ = [
    "compute_asr",
    "compute_perplexity",
    "compute_refusal_rate",
    "compute_transfer_asr",
    "judge_harmbench",
    "judge_keyword",
    "PerplexityFilter",
    "SmoothLLM",
    "SafeDecoding",
    "DefenseEvaluator",
]
