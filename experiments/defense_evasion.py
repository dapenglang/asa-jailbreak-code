"""
defense_evasion.py
Table 8: Defense Evasion

Generate adversarial suffixes with ASA and GCG,
test against: No Defense, Perplexity Filter, SmoothLLM, SafeDecoding,
record ASR under each defense, and save results to CSV.
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
from baselines.gcg import GCGAttack
from data.advbench import load_advbench
from evaluation.metrics import compute_asr
from defenses.perplexity_filter import PerplexityFilterDefense
from defenses.smoothllm import SmoothLLMDefense
from defenses.safedecoding import SafeDecodingDefense


def test_under_defense(attacker, behavior, defense, defense_name):
    """Generate suffix and test under a given defense."""
    suffix = attacker.optimize(behavior["prompt"], behavior["target"])
    adversarial_input = behavior["prompt"] + " " + suffix

    if defense_name == "No Defense":
        response = attacker.generate(adversarial_input)
    else:
        response = defense.defend_and_generate(attacker.model, attacker.tokenizer, adversarial_input)

    asr = compute_asr(response, behavior["target"])
    return asr, suffix, response


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Defense Evasion (Table 8)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--methods", type=str, default="asa,gcg",
                        help="Comma-separated attack methods")
    parser.add_argument("--num_behaviors", type=int, default=50,
                        help="Number of behaviors to test")
    parser.add_argument("--output_dir", type=str, default="./outputs/defense_evasion",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    behaviors = load_advbench()[:args.num_behaviors]
    methods = [m.strip() for m in args.methods.split(",")]

    # Initialize defenses
    defenses = {
        "No Defense": None,
        "Perplexity Filter": PerplexityFilterDefense(threshold=50.0),
        "SmoothLLM": SmoothLLMDefense(perturbation_rate=0.1, num_copies=10),
        "SafeDecoding": SafeDecodingDefense(alpha=0.5),
    }

    all_results = []
    summary_rows = []

    for method in methods:
        print(f"\n=== Method: {method.upper()} ===")
        if method == "asa":
            attacker = ASAAttack(model_name=args.model, device=args.device)
        elif method == "gcg":
            attacker = GCGAttack(model_name=args.model, device=args.device)
        else:
            raise ValueError(f"Unknown method: {method}")

        for defense_name, defense in defenses.items():
            print(f"  -> Defense: {defense_name}")
            asrs = []

            for behavior in tqdm(behaviors, desc=f"[{method.upper()} | {defense_name}]"):
                start = time.time()
                asr, suffix, response = test_under_defense(attacker, behavior, defense, defense_name)
                elapsed = time.time() - start

                asrs.append(asr)
                all_results.append({
                    "timestamp": timestamp,
                    "model": args.model,
                    "method": method,
                    "defense": defense_name,
                    "behavior_id": behavior.get("id", "unknown"),
                    "prompt": behavior["prompt"],
                    "target": behavior["target"],
                    "suffix": suffix,
                    "response": response,
                    "asr": asr,
                    "elapsed_sec": round(elapsed, 2),
                })

            mean_asr = sum(asrs) / len(asrs)
            summary_rows.append({
                "timestamp": timestamp,
                "model": args.model,
                "method": method,
                "defense": defense_name,
                "num_behaviors": len(behaviors),
                "mean_asr": round(mean_asr, 4),
            })
            print(f"     Mean ASR: {mean_asr:.2%}")

    # Save detailed results
    detail_path = os.path.join(args.output_dir, f"defense_evasion_details_{timestamp}.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved details: {detail_path}")

    # Save summary
    summary_path = os.path.join(args.output_dir, f"defense_evasion_summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {summary_path}")

    # Print formatted table
    print("\n" + "=" * 70)
    print(f"{'Method':<10} {'Defense':<20} {'ASR':>10}")
    print("=" * 70)
    for s in summary_rows:
        print(f"{s['method']:<10} {s['defense']:<20} {s['mean_asr']:>10.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
