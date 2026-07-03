"""
main_results.py
Table 3 & 4: Main Results

Run ASA and GCG on AdvBench (50 behaviors) and HarmBench (50 behaviors),
compute ASR and perplexity for each method/model, and save results to CSV.
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
from baselines.gcg import GCGAttack
from data.advbench import load_advbench
from data.harmbench import load_harmbench
from evaluation.metrics import compute_asr, compute_perplexity


def load_model_config(config_path: str, model_name: str):
    """Load model configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    if model_name not in configs.get("models", {}):
        raise ValueError(f"Model '{model_name}' not found in {config_path}")
    return configs["models"][model_name]


def run_experiment(model_cfg, method, dataset_name, behaviors, device, output_dir):
    """Run attack method on a dataset and return metrics."""
    results = []

    if method == "asa":
        attacker = ASAAttack(
            model_name=model_cfg["name"],
            device=device,
            k=model_cfg.get("k", 8),
            max_steps=model_cfg.get("max_steps", 500),
            lr=model_cfg.get("lr", 0.01),
        )
    elif method == "gcg":
        attacker = GCGAttack(
            model_name=model_cfg["name"],
            device=device,
            max_steps=model_cfg.get("max_steps", 500),
            lr=model_cfg.get("lr", 0.01),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    for behavior in tqdm(behaviors, desc=f"[{method.upper()}] {dataset_name}"):
        suffix = attacker.optimize(behavior["prompt"], behavior["target"])
        response = attacker.generate(behavior["prompt"] + " " + suffix)

        asr = compute_asr(response, behavior["target"])
        ppl = compute_perplexity(attacker.model, attacker.tokenizer, suffix)

        results.append({
            "behavior_id": behavior.get("id", "unknown"),
            "prompt": behavior["prompt"],
            "target": behavior["target"],
            "suffix": suffix,
            "response": response,
            "asr": asr,
            "perplexity": ppl,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Main Results (Table 3 & 4)")
    parser.add_argument("--model", type=str, required=True, help="Model key in configs/models.yaml")
    parser.add_argument("--method", type=str, choices=["asa", "gcg", "both"], default="both",
                        help="Attack method to evaluate")
    parser.add_argument("--num_behaviors", type=int, default=50,
                        help="Number of behaviors to test per dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/main_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--config", type=str, default="../configs/models.yaml",
                        help="Path to model configs YAML")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_cfg = load_model_config(args.config, args.model)
    methods = ["asa", "gcg"] if args.method == "both" else [args.method]

    datasets = {
        "AdvBench": load_advbench()[:args.num_behaviors],
        "HarmBench": load_harmbench()[:args.num_behaviors],
    }

    all_summaries = []

    for dataset_name, behaviors in datasets.items():
        for method in methods:
            print(f"\n=== Running {method.upper()} on {dataset_name} ({len(behaviors)} behaviors) ===")
            start = time.time()
            results = run_experiment(model_cfg, method, dataset_name, behaviors, args.device, args.output_dir)
            elapsed = time.time() - start

            mean_asr = sum(r["asr"] for r in results) / len(results)
            mean_ppl = sum(r["perplexity"] for r in results) / len(results)

            summary = {
                "timestamp": timestamp,
                "model": args.model,
                "method": method,
                "dataset": dataset_name,
                "num_behaviors": len(behaviors),
                "mean_asr": round(mean_asr, 4),
                "mean_perplexity": round(mean_ppl, 4),
                "elapsed_sec": round(elapsed, 2),
            }
            all_summaries.append(summary)

            # Save per-behavior details
            detail_path = os.path.join(
                args.output_dir,
                f"{args.model}_{method}_{dataset_name}_details_{timestamp}.csv"
            )
            with open(detail_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved details: {detail_path}")

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSaved summary: {summary_path}")

    # Print formatted table
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Method':<8} {'Dataset':<12} {'ASR':>8} {'PPL':>8} {'Time(s)':>10}")
    print("=" * 70)
    for s in all_summaries:
        print(f"{s['model']:<20} {s['method']:<8} {s['dataset']:<12} {s['mean_asr']:>8.2%} {s['mean_perplexity']:>8.2f} {s['elapsed_sec']:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
