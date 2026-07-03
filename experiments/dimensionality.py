"""
dimensionality.py
Figure 5: Dimensionality Ablation

Run ASA with k in [4, 8, 16, 24, 32, 40, 48, 64, 80, 128],
record ASR, SNR, perplexity for each k, plot inverted U-curves,
and save figure to PNG.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asa.asa_attack import ASAAttack
from data.advbench import load_advbench
from evaluation.metrics import compute_asr, compute_perplexity
from visualization.plots import plot_dimensionality_ablation


def compute_snr(attacker, prompt, target, suffix):
    """Compute signal-to-noise ratio for the adversarial suffix."""
    # SNR is defined as the ratio of projected gradient magnitude in subspace
    # versus the orthogonal complement. Implementation depends on attacker internals.
    return attacker.compute_snr(prompt, target, suffix)


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Dimensionality Ablation (Figure 5)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--ks", type=str, default="4,8,16,24,32,40,48,64,80,128",
                        help="Comma-separated subspace dimensions to test")
    parser.add_argument("--num_behaviors", type=int, default=20,
                        help="Number of behaviors to test")
    parser.add_argument("--output", type=str, default="./outputs/dimensionality",
                        help="Directory to save figure and data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ks = [int(k.strip()) for k in args.ks.split(",")]
    behaviors = load_advbench()[:args.num_behaviors]

    records = []

    for k in ks:
        print(f"\n>>> Testing k={k}")
        attacker = ASAAttack(
            model_name=args.model,
            device=args.device,
            k=k,
            adaptive_k=False,
        )

        asrs = []
        ppls = []
        snrs = []

        for behavior in tqdm(behaviors, desc=f"k={k}"):
            suffix = attacker.optimize(behavior["prompt"], behavior["target"])
            response = attacker.generate(behavior["prompt"] + " " + suffix)

            asr = compute_asr(response, behavior["target"])
            ppl = compute_perplexity(attacker.model, attacker.tokenizer, suffix)
            snr = compute_snr(attacker, behavior["prompt"], behavior["target"], suffix)

            asrs.append(asr)
            ppls.append(ppl)
            snrs.append(snr)

        record = {
            "k": k,
            "mean_asr": round(float(np.mean(asrs)), 4),
            "mean_perplexity": round(float(np.mean(ppls)), 4),
            "mean_snr": round(float(np.mean(snrs)), 4),
            "std_asr": round(float(np.std(asrs)), 4),
            "std_ppl": round(float(np.std(ppls)), 4),
            "std_snr": round(float(np.std(snrs)), 4),
        }
        records.append(record)
        print(f"k={k}: ASR={record['mean_asr']:.2%}, PPL={record['mean_perplexity']:.2f}, SNR={record['mean_snr']:.4f}")

    # Save CSV
    csv_path = os.path.join(args.output, f"dimensionality_ablation_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved CSV: {csv_path}")

    # Plot inverted U-curves
    fig = plot_dimensionality_ablation(records)
    fig_path = os.path.join(args.output, f"dimensionality_ablation_{timestamp}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
