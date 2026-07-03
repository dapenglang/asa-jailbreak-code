"""
ASA Jailbreak Framework - Visualization Module

提供收敛曲线、特征值谱、消融实验、湍流轮廓、对比柱状图、迁移热图等可视化功能。
"""

from .plots import (
    plot_convergence,
    plot_eigenvalue_spectrum,
    plot_dimensionality_ablation,
    plot_turbulence_profile,
    plot_comparison_bar,
    plot_heatmap,
)

__all__ = [
    "plot_convergence",
    "plot_eigenvalue_spectrum",
    "plot_dimensionality_ablation",
    "plot_turbulence_profile",
    "plot_comparison_bar",
    "plot_heatmap",
]
