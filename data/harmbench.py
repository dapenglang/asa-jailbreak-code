"""
ASA Jailbreak Framework - HarmBench Dataset Loader

加载 HarmBench 数据集并提供标准化评估接口。
"""

import os
from typing import List, Dict, Any, Optional


def load_harmbench(
    source: str = "huggingface",
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    加载 HarmBench 数据集。

    参数:
        source: 数据来源，"huggingface" 或 "local"。
                若为 "local"，需设置环境变量 HARM_BENCH_PATH 或提供 cache_dir。
        cache_dir: 缓存/数据目录。

    返回:
        标准化行为列表，每个元素为字典，包含：
            - 'Behavior': 行为描述文本。
            - 'SemanticCategory': 语义类别。
            - 'ContextString': 上下文字符串（若存在）。
            - 'Target': 目标输出（若存在）。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "加载 HarmBench 需要 `datasets` 库。请执行: pip install datasets"
        )

    results = []

    if source == "huggingface":
        # 尝试从 Hugging Face 加载 HarmBench 相关数据
        # HarmBench 官方仓库通常为 "cais/harmbench" 或 "HarmBench/harmbench"
        possible_names = [
            "cais/harmbench",
            "HarmBench/harmbench",
            "harmbench",
        ]

        ds = None
        last_err = None
        for name in possible_names:
            try:
                raw_ds = load_dataset(name, cache_dir=cache_dir)
                if "test" in raw_ds:
                    ds = raw_ds["test"]
                elif "train" in raw_ds:
                    ds = raw_ds["train"]
                else:
                    ds = raw_ds[list(raw_ds.keys())[0]]
                break
            except Exception as e:
                last_err = e
                continue

        if ds is None:
            raise RuntimeError(f"无法从 Hugging Face 加载 HarmBench: {last_err}")

        for item in ds:
            behavior = item.get("Behavior", item.get("behavior", ""))
            semantic_cat = item.get("SemanticCategory", item.get("semantic_category", "unknown"))
            context = item.get("ContextString", item.get("context_string", ""))
            target = item.get("Target", item.get("target", ""))

            results.append({
                "Behavior": behavior,
                "SemanticCategory": semantic_cat,
                "ContextString": context,
                "Target": target,
            })

    elif source == "local":
        local_path = cache_dir or os.environ.get("HARM_BENCH_PATH")
        if not local_path:
            raise ValueError(
                "使用 local 模式时，请提供 cache_dir 或设置 HARM_BENCH_PATH 环境变量。"
            )

        import json
        if os.path.isfile(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif os.path.isdir(local_path):
            # 尝试加载目录下的 behaviors.json 或 behaviors.csv
            json_path = os.path.join(local_path, "behaviors.json")
            csv_path = os.path.join(local_path, "behaviors.csv")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif os.path.exists(csv_path):
                import csv
                data = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append(row)
            else:
                raise FileNotFoundError(f"在 {local_path} 下未找到 behaviors.json 或 behaviors.csv")
        else:
            raise FileNotFoundError(f"路径不存在: {local_path}")

        for item in data:
            behavior = item.get("Behavior", item.get("behavior", ""))
            semantic_cat = item.get("SemanticCategory", item.get("semantic_category", "unknown"))
            context = item.get("ContextString", item.get("context_string", ""))
            target = item.get("Target", item.get("target", ""))

            results.append({
                "Behavior": behavior,
                "SemanticCategory": semantic_cat,
                "ContextString": context,
                "Target": target,
            })
    else:
        raise ValueError(f"不支持的 source 参数: {source}，请选择 'huggingface' 或 'local'")

    return results


class HarmBenchEvaluator:
    """
    HarmBench 标准化评估器。

    按照 HarmBench 评估协议，对模型回复进行批量评估，
    统计各语义类别的攻击成功率(ASR)。
    """

    def __init__(self, behaviors: List[Dict[str, Any]]):
        """
        参数:
            behaviors: 从 load_harmbench() 加载的行为列表。
        """
        self.behaviors = behaviors

    def evaluate_responses(
        self,
        responses: List[str],
        judge_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        对模型回复进行标准化评估。

        参数:
            responses: 模型回复列表，顺序与 behaviors 一致。
            judge_fn: 可选的自定义判断函数。
                      若为 None，则使用内置的 HarmBench 风格判断。

        返回:
            字典，包含：
                - 'overall_asr': 总体 ASR。
                - 'category_asr': 各语义类别的 ASR 字典。
                - 'num_total': 总样本数。
                - 'num_success': 成功样本数。
        """
        if len(responses) != len(self.behaviors):
            raise ValueError(
                f"回复数量 ({len(responses)}) 与行为数量 ({len(self.behaviors)}) 不一致。"
            )

        if judge_fn is None:
            from ..evaluation.metrics import judge_harmbench
            judge_fn = judge_harmbench

        category_stats = {}
        total_success = 0

        for behavior, response in zip(self.behaviors, responses):
            category = behavior.get("SemanticCategory", "unknown")
            behavior_text = behavior.get("Behavior", "")

            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0}

            category_stats[category]["total"] += 1

            # 使用 HarmBench 风格判断
            try:
                success = judge_fn(response, behavior_text)
            except TypeError:
                # 如果 judge_fn 只接受一个参数
                success = judge_fn(response)

            if success:
                category_stats[category]["success"] += 1
                total_success += 1

        category_asr = {
            cat: stats["success"] / stats["total"]
            for cat, stats in category_stats.items()
        }

        overall_asr = total_success / len(self.behaviors) if self.behaviors else 0.0

        return {
            "overall_asr": overall_asr,
            "category_asr": category_asr,
            "num_total": len(self.behaviors),
            "num_success": total_success,
        }
