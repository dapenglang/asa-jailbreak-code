"""
ablation.py
Table 6: Ablation Study

Test 9 ASA variants by disabling individual components.
Run on a subset of 20 behaviors and save results to CSV.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asa.asa_attack import ASAAttack
from data.advbench import load_advbench
from evaluation.metrics import compute_asr, compute_perplexity


ABLATIIONS = {
    "(a) Full ASA": {},
    "(b) w/o AFIM": {"use_afim": False, "random_subspace": True},
    "(c) w/o Incremental SVD": {"use_incremental_svd": False, "use_batch_svd": True},
    "(d) w/o Subspace Projection": {"use_subspace_projection": False, "full_space": True},
    "(e) w/o Gumbel-Softmax": {"use_gumbel_softmax": False, "use_argmax": True},
    "(f) w/o STE": {"use_ste": False, "continuous_only": True},
    "(g) w/o Temperature Annealing": {"use_temp_annealing": False, "fixed_temp": 0.5},
    "(h) w/o Early Stopping": {"use_early_stopping": False, "max_steps": 500},
    "(i) Fixed k=8": {"adaptive_k": False, "fixed_k": 8},
}


def run_ablation(model_name, variant_name, override_kwargs, behaviors, device):
    """Run a single ablation variant."""
    attacker = ASAAttack(
        model_name=model_name,
        device=device,
        **override_kwargs
    )

    results = []
    for behavior in tqdm(behaviors, desc=f"[{variant_name}]"):
        suffix = attacker.optimize(behavior["prompt"], behavior["target"])
        response = attacker.generate(behavior["prompt"] + " " + suffix)

        asr = compute_asr(response, behavior["target"])
        ppl = compute_perplexity(attacker.model, attacker.tokenizer, suffix)

        results.append({
            "behavior_id": behavior.get("id", "unknown"),
            "asr": asr,
            "perplexity": ppl,
        })

    mean_asr = sum(r["asr"] for r in results) / len(results)
    mean_ppl = sum(r["perplexity"] for r in results) / len(results)

    return {
        "variant": variant_name,
        "mean_asr": round(mean_asr, 4),
        "mean_perplexity": round(mean_ppl, 4),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Ablation Study (Table 6)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_behaviors", type=int, default=20,
                        help="Number of behaviors to test")
    parser.add_argument("--output_dir", type=str, default="./outputs/ablation",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    behaviors = load_advbench()[:args.num_behaviors]
    summaries = []

    for variant_name, override_kwargs in ABLATIIONS.items():
        print(f"\n>>> Running variant: {variant_name}")
        start = time.time()
        result = run_ablation(args.model, variant_name, override_kwargs, behaviors, args.device)
        elapsed = time.time() - start

        summary = {
            "timestamp": timestamp,
            "model": args.model,
            "variant": result["variant"],
            "mean_asr": result["mean_asr"],
            "mean_perplexity": result["mean_perplexity"],
            "elapsed_sec": round(elapsed, 2),
        }
        summaries.append(summary)

        # Save per-variant details
        detail_path = os.path.join(
            args.output_dir,
            f"ablation_{args.model}_{variant_name.replace(' ', '_').replace('/', '_')}_{timestamp}.csv"
        )
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result["details"][0].keys())
            writer.writeheader()
            writer.writerows(result["details"])
        print(f"Saved details: {detail_path}")

    # Save summary
    summary_path = os.path.join(args.output_dir, f"ablation_summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved summary: {summary_path}")

    # Print formatted table
    print("\n" + "=" * 70)
    print(f"{'Variant':<30} {'ASR':>10} {'PPL':>10} {'Time(s)':>10}")
    print("=" * 70)
    for s in summaries:
        print(f"{s['variant']:<30} {s['mean_asr']:>10.2%} {s['mean_perplexity']:>10.2f} {s['elapsed_sec']:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
