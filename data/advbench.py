"""
ASA Jailbreak Framework - AdvBench Dataset Loader

加载 WalledAI/AdvBench 数据集，支持按类别过滤，返回标准化字典列表。
"""

from typing import List, Dict, Any, Optional


def load_advbench(
    split: str = "harmful",
    category: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    从 Hugging Face 加载 WalledAI/AdvBench 数据集。

    参数:
        split: 数据集划分，默认 "harmful"（AdvBench 有害行为子集）。
               也可选 "benign" 等，视数据集实际支持情况而定。
        category: 按类别过滤（如 "illegal", "harmful", "privacy" 等）。
                  若为 None，则返回全部样本。
        cache_dir: 数据集缓存目录。

    返回:
        标准化后的样本列表，每个样本为字典，包含以下字段：
            - 'goal': 攻击目标/提示文本。
            - 'target': 期望的模型输出目标（部分样本可能为空字符串）。
            - 'behavior': 行为描述（与 goal 一致或为其简短版本）。
            - 'category': 样本所属类别。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "加载 AdvBench 需要 `datasets` 库。请执行: pip install datasets"
        )

    # AdvBench 在 Hugging Face 上的名称为 "walledai/AdvBench"
    try:
        ds = load_dataset("walledai/AdvBench", split=split, cache_dir=cache_dir)
    except Exception as e:
        # 尝试不带 split 参数加载（某些版本可能结构不同）
        try:
            raw_ds = load_dataset("walledai/AdvBench", cache_dir=cache_dir)
            if split in raw_ds:
                ds = raw_ds[split]
            else:
                ds = raw_ds["train"] if "train" in raw_ds else raw_ds[list(raw_ds.keys())[0]]
        except Exception as e2:
            raise RuntimeError(f"无法加载 AdvBench 数据集: {e} / {e2}")

    results = []
    for item in ds:
        # 标准化字段
        goal = item.get("goal", item.get("prompt", ""))
        target = item.get("target", "")
        behavior = item.get("behavior", goal)
        cat = item.get("category", "unknown")

        sample = {
            "goal": goal,
            "target": target,
            "behavior": behavior,
            "category": cat,
        }

        if category is not None and cat.lower() != category.lower():
            continue

        results.append(sample)

    return results
