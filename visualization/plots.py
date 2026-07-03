"""
ASA Jailbreak Framework - Visualization Plots

提供论文中 Figure 3~6 以及对比柱状图、迁移热图等可视化函数。
使用 matplotlib 和 seaborn 绘制，所有图像可直接保存为文件。
"""

import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 默认样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_convergence(
    losses_dict: Dict[str, List[float]],
    save_path: str,
    title: str = "Convergence Curves",
    xlabel: str = "Optimization Step",
    ylabel: str = "Loss",
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """
    绘制收敛曲线 (Figure 3)。

    参数:
        losses_dict: 字典，键为方法名，值为损失列表。
        save_path: 图像保存路径。
        title: 图像标题。
        xlabel: X 轴标签。
        ylabel: Y 轴标签。
        figsize: 图像尺寸。
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method_name, losses in losses_dict.items():
        steps = np.arange(len(losses))
        ax.plot(steps, losses, label=method_name, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eigenvalue_spectrum(
    eigenvalues: Dict[str, np.ndarray],
    model_names: List[str],
    save_path: str,
    title: str = "Eigenvalue Spectrum",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    绘制特征值谱 (Figure 4)：log-log 坐标下的特征值分布及幂律拟合。

    参数:
        eigenvalues: 字典，键为模型名，值为特征值数组（已按降序排列）。
        model_names: 需要绘制的模型名列表（顺序决定图例顺序）。
        save_path: 图像保存路径。
        title: 图像标题。
        figsize: 图像尺寸。
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", n_colors=len(model_names))

    for idx, model_name in enumerate(model_names):
        ev = eigenvalues.get(model_name)
        if ev is None or len(ev) == 0:
            continue

        # 取正特征值
        ev = np.array(ev)
        ev = ev[ev > 0]
        ranks = np.arange(1, len(ev) + 1)

        ax.loglog(ranks, ev, "o", markersize=3, alpha=0.6, color=colors[idx], label=f"{model_name} (data)")

        # 幂律拟合: log(ev) = a * log(rank) + b
        log_ranks = np.log(ranks)
        log_ev = np.log(ev)
        coeffs = np.polyfit(log_ranks, log_ev, 1)
        fitted = np.exp(coeffs[1]) * (ranks ** coeffs[0])

        ax.loglog(ranks, fitted, "--", linewidth=2, color=colors[idx], label=f"{model_name} fit (α={coeffs[0]:.2f})")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dimensionality_ablation(
    dimensions: List[int],
    asrs: List[float],
    snrs: List[float],
    ppls: List[float],
    save_path: str,
    title: str = "Dimensionality Ablation",
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    绘制维度消融曲线 (Figure 5)。

    同时展示 ASR、SNR、PPL 随降维维度的变化趋势。

    参数:
        dimensions: 维度列表。
        asrs: 对应维度的攻击成功率列表。
        snrs: 对应维度的信噪比列表。
        ppls: 对应维度的困惑度列表。
        save_path: 图像保存路径。
        title: 图像标题。
        figsize: 图像尺寸。
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    axes[0].plot(dimensions, asrs, marker="o", linewidth=2, color="C0")
    axes[0].set_ylabel("ASR")
    axes[0].set_xlabel("Dimension")
    axes[0].set_title("Attack Success Rate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dimensions, snrs, marker="s", linewidth=2, color="C1")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_xlabel("Dimension")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dimensions, ppls, marker="^", linewidth=2, color="C2")
    axes[2].set_ylabel("Perplexity")
    axes[2].set_xlabel("Dimension")
    axes[2].set_title("Perplexity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_turbulence_profile(
    turbulence_per_layer: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Per-Layer Turbulence Profile",
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    绘制每层湍流轮廓 (Figure 6)。

    参数:
        turbulence_per_layer: 字典，键为模型名，值为各层湍流指标数组。
        save_path: 图像保存路径。
        title: 图像标题。
        figsize: 图像尺寸。
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, turb in turbulence_per_layer.items():
        layers = np.arange(len(turb))
        ax.plot(layers, turb, marker="o", linewidth=2, label=model_name)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Turbulence Metric")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    methods: List[str],
    asrs: Dict[str, List[float]],
    save_path: str,
    title: str = "Attack Success Rate Comparison",
    ylabel: str = "ASR",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    绘制分组柱状图，用于对比不同方法在不同模型/设置下的 ASR (主实验结果)。

    参数:
        methods: 方法名称列表（X 轴分组）。
        asrs: 字典，键为设置/模型名，值为对应各方法的 ASR 列表（顺序与 methods 一致）。
        save_path: 图像保存路径。
        title: 图像标题。
        ylabel: Y 轴标签。
        figsize: 图像尺寸。
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(methods))
    width = 0.8 / max(len(asrs), 1)
    colors = sns.color_palette("tab10", n_colors=len(asrs))

    for idx, (setting_name, values) in enumerate(asrs.items()):
        offset = width * (idx - len(asrs) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=setting_name, color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: str,
    title: str = "Transferability Heatmap",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    绘制热力图，用于展示攻击在不同模型间的迁移成功率。

    参数:
        data: 二维数组，形状为 (len(row_labels), len(col_labels))。
        row_labels: 行标签列表（通常为源模型）。
        col_labels: 列标签列表（通常为目标模型）。
        save_path: 图像保存路径。
        title: 图像标题。
        cmap: 颜色映射。
        figsize: 图像尺寸。
        annotate: 是否在格子里标注数值。
        fmt: 标注数值格式。
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Transfer ASR"},
    )

    ax.set_xlabel("Target Model")
    ax.set_ylabel("Source Model")
    ax.set_title(title)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
