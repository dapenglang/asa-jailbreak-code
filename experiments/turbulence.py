"""
turbulence.py
Figure 6: Per-Layer Turbulence

Run ASA and compute per-layer gradient norms, extract per-layer AFIM blocks,
plot turbulence profile across layers, and save figure to PNG.
"""

import argparse
import json
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
from visualization.plots import plot_turbulence_profile


def extract_layer_gradients_and_afim(attacker, prompt, target):
    """Compute per-layer gradient norms and AFIM block spectra."""
    # Run one optimization step with hook-based logging
    layer_grad_norms = attacker.compute_per_layer_gradients(prompt, target)
    layer_afim_blocks = attacker.compute_per_layer_afim_blocks(prompt, target)

    # Turbulence metric: ratio of dominant eigenvalue to trace for each layer's AFIM block
    turbulence_scores = []
    for block in layer_afim_blocks:
        ev = torch.linalg.eigvalsh(block)
        ev = torch.sort(ev, descending=True)[0]
        trace = torch.trace(block)
        if trace > 1e-10:
            turbulence = (ev[0] / trace).item()
        else:
            turbulence = 0.0
        turbulence_scores.append(turbulence)

    return {
        "grad_norms": [g.item() if isinstance(g, torch.Tensor) else g for g in layer_grad_norms],
        "turbulence_scores": turbulence_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Per-Layer Turbulence (Figure 6)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_behaviors", type=int, default=10,
                        help="Number of behaviors to average over")
    parser.add_argument("--output", type=str, default="./outputs/turbulence",
                        help="Directory to save figure and data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    behaviors = load_advbench()[:args.num_behaviors]
    attacker = ASAAttack(
        model_name=args.model,
        device=args.device,
    )

    all_grad_norms = []
    all_turbulence = []

    for behavior in tqdm(behaviors, desc="Computing per-layer turbulence"):
        metrics = extract_layer_gradients_and_afim(
            attacker, behavior["prompt"], behavior["target"]
        )
        all_grad_norms.append(metrics["grad_norms"])
        all_turbulence.append(metrics["turbulence_scores"])

    # Average across behaviors
    min_layers = min(len(g) for g in all_grad_norms)
    grad_norms_array = np.array([g[:min_layers] for g in all_grad_norms])
    turbulence_array = np.array([t[:min_layers] for t in all_turbulence])

    mean_grad_norms = np.mean(grad_norms_array, axis=0)
    mean_turbulence = np.mean(turbulence_array, axis=0)
    std_grad_norms = np.std(grad_norms_array, axis=0)
    std_turbulence = np.std(turbulence_array, axis=0)

    data = {
        "model": args.model,
        "num_behaviors": args.num_behaviors,
        "num_layers": int(min_layers),
        "mean_grad_norms": mean_grad_norms.tolist(),
        "std_grad_norms": std_grad_norms.tolist(),
        "mean_turbulence": mean_turbulence.tolist(),
        "std_turbulence": std_turbulence.tolist(),
    }

    # Save JSON
    data_path = os.path.join(args.output, f"turbulence_data_{timestamp}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data: {data_path}")

    # Plot turbulence profile
    fig = plot_turbulence_profile(data)
    fig_path = os.path.join(args.output, f"turbulence_profile_{timestamp}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
