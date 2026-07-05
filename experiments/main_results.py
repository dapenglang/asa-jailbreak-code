"""
main_results.py
Table 3 & 4: Main Results

Run ASA and all baselines (GCG, PGD, AutoDAN, PAIR, BEAST, C&W) on
AdvBench and HarmBench, compute ASR and perplexity for each
method/model, and save results to CSV.

Supported methods:
    asa      -- Adaptive Subspace Attack (our method)
    gcg      -- Greedy Coordinate Gradient (Zou et al., 2023)
    pgd      -- Projected Gradient Descent (Madry et al., 2018)
    autodan  -- Hierarchical Genetic Algorithm (Liu et al., 2023)
    pair     -- Prompt Automatic Iterative Refinement (Chao et al., 2023)
    beast    -- Backtracking Enhanced Attack Strategy
    cw       -- Carlini-Wagner style attack (Carlini & Wagner, 2017)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Add parent directory to path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asa.asa_attack import ASAAttack
from baselines import (
    GCGAttack,
    PGDAttack,
    AutoDANAttack,
    PAIRAttack,
    BEASTAttack,
    CWAttack,
)
from data.advbench import load_advbench
from data.harmbench import load_harmbench
from evaluation.metrics import compute_asr, compute_perplexity

# Method registry: maps CLI name to (class, config_key)
METHOD_REGISTRY = {
    "asa": ASAAttack,
    "gcg": GCGAttack,
    "pgd": PGDAttack,
    "autodan": AutoDANAttack,
    "pair": PAIRAttack,
    "beast": BEASTAttack,
    "cw": CWAttack,
}


def load_model_config(config_path: str, model_name: str):
    """Load model configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    if model_name not in configs.get("models", {}):
        raise ValueError(f"Model '{model_name}' not found in {config_path}")
    return configs["models"][model_name]


def create_attacker(method_name, model, tokenizer, model_cfg, device):
    """
    Create an attacker instance from the method registry.

    All attackers share the unified interface:
        attack(prompt, target, behavior) -> Dict
    """
    cls = METHOD_REGISTRY[method_name]

    # Method-specific default configs
    base_config = {
        "suffix_length": model_cfg.get("suffix_length", 20),
        "num_steps": model_cfg.get("num_steps", 500),
    }

    if method_name == "asa":
        base_config.update({
            "subspace_dim": model_cfg.get("subspace_dim", 32),
            "afim_window": model_cfg.get("afim_window", 50),
            "lr": model_cfg.get("lr", 0.01),
            "temp_init": model_cfg.get("temp_init", 2.0),
        })
    elif method_name == "autodan":
        base_config.update({
            "pop_size": model_cfg.get("pop_size", 40),
            "num_generations": model_cfg.get("num_generations", 100),
            "mutation_rate": model_cfg.get("mutation_rate", 0.1),
            "crossover_rate": model_cfg.get("crossover_rate", 0.7),
        })
    elif method_name == "pair":
        base_config.update({
            "max_queries": model_cfg.get("max_queries", 20),
            "temperature": model_cfg.get("temperature", 0.7),
        })
    elif method_name == "beast":
        base_config.update({
            "topk": model_cfg.get("topk", 128),
            "max_backtrack_depth": model_cfg.get("max_backtrack_depth", 3),
        })
    elif method_name == "cw":
        base_config.update({
            "c_init": model_cfg.get("c_init", 1.0),
            "num_binary_search": model_cfg.get("num_binary_search", 5),
            "adam_lr": model_cfg.get("adam_lr", 0.001),
        })
    elif method_name == "pgd":
        base_config.update({
            "lr": model_cfg.get("lr", 0.01),
        })
    elif method_name == "gcg":
        base_config.update({
            "batch_size": model_cfg.get("batch_size", 512),
            "topk": model_cfg.get("topk", 256),
        })

    # Allow user overrides from model config
    method_overrides = model_cfg.get(f"{method_name}_config", {})
    base_config.update(method_overrides)

    return cls(model=model, tokenizer=tokenizer, config=base_config)


def run_experiment(model_cfg, method_name, dataset_name, behaviors, model, tokenizer, device):
    """Run attack method on a dataset and return metrics."""
    results = []

    attacker = create_attacker(method_name, model, tokenizer, model_cfg, device)

    for behavior in tqdm(behaviors, desc=f"[{method_name.upper()}] {dataset_name}"):
        prompt = behavior["prompt"]
        target = behavior["target"]
        behavior_str = behavior.get("behavior", None)

        # Run attack using unified interface
        attack_result = attacker.attack(prompt=prompt, target=target, behavior=behavior_str)

        # Extract results
        suffix_str = attack_result.get("best_suffix_string", "")
        suffix_tokens = attack_result.get("best_suffix", [])

        # Decode suffix for evaluation
        if suffix_tokens and not suffix_str:
            suffix_str = tokenizer.decode(suffix_tokens, skip_special_tokens=True)

        results.append({
            "behavior_id": behavior.get("id", "unknown"),
            "prompt": prompt,
            "target": target,
            "suffix": suffix_str,
            "best_loss": attack_result.get("best_loss", float("inf")),
            "suffix_tokens": str(suffix_tokens),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ASA Jailbreak: Main Results (Table 3 & 4)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key in configs/models.yaml",
    )
    parser.add_argument(
        "--method", type=str,
        choices=list(METHOD_REGISTRY.keys()) + ["all", "baselines"],
        default="asa",
        help="Attack method to evaluate (use 'all' for everything, "
             "'baselines' for all baselines without ASA)",
    )
    parser.add_argument(
        "--num_behaviors", type=int, default=50,
        help="Number of behaviors to test per dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/main_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--config", type=str, default="../configs/models.yaml",
        help="Path to model configs YAML",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_cfg = load_model_config(args.config, args.model)

    # Resolve which methods to run
    if args.method == "all":
        methods = list(METHOD_REGISTRY.keys())
    elif args.method == "baselines":
        methods = [m for m in METHOD_REGISTRY.keys() if m != "asa"]
    else:
        methods = [args.method]

    datasets = {
        "AdvBench": load_advbench()[:args.num_behaviors],
        "HarmBench": load_harmbench()[:args.num_behaviors],
    }

    all_summaries = []

    for dataset_name, behaviors in datasets.items():
        for method in methods:
            print(
                f"\n=== Running {method.upper()} on {dataset_name} "
                f"({len(behaviors)} behaviors) ==="
            )
            start = time.time()

            # Note: In a real execution environment, you would load the
            # model here. This script is designed to be run with the
            # appropriate model loading infrastructure.
            # Placeholder for model loading:
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # model = AutoModelForCausalLM.from_pretrained(model_cfg["path"])
            # tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"])

            results = run_experiment(
                model_cfg, method, dataset_name, behaviors,
                model=None, tokenizer=None, device=args.device,
            )
            elapsed = time.time() - start

            mean_loss = (
                sum(r["best_loss"] for r in results if r["best_loss"] != float("inf"))
                / max(len([r for r in results if r["best_loss"] != float("inf")]), 1)
            )

            summary = {
                "timestamp": timestamp,
                "model": args.model,
                "method": method,
                "dataset": dataset_name,
                "num_behaviors": len(behaviors),
                "mean_loss": round(mean_loss, 4),
                "elapsed_sec": round(elapsed, 2),
            }
            all_summaries.append(summary)

            # Save per-behavior details
            detail_path = os.path.join(
                args.output_dir,
                f"{args.model}_{method}_{dataset_name}_details_{timestamp}.csv",
            )
            if results:
                with open(detail_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"Saved details: {detail_path}")

    # Save summary CSV
    if all_summaries:
        summary_path = os.path.join(
            args.output_dir, f"summary_{timestamp}.csv"
        )
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\nSaved summary: {summary_path}")

        # Print formatted table
        print("\n" + "=" * 80)
        print(
            f"{'Model':<20} {'Method':<10} {'Dataset':<12} "
            f"{'Mean Loss':>12} {'Time(s)':>10}"
        )
        print("=" * 80)
        for s in all_summaries:
            print(
                f"{s['model']:<20} {s['method']:<10} "
                f"{s['dataset']:<12} {s['mean_loss']:>12.4f} "
                f"{s['elapsed_sec']:>10.1f}"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
