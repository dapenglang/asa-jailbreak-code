"""
convergence.py
Figure 3: Convergence Curves

Run ASA and GCG on the same 5 behaviors, record loss at each step,
plot convergence curves, and save figure to PNG.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asa.asa_attack import ASAAttack
from baselines.gcg import GCGAttack
from data.advbench import load_advbench
from visualization.plots import plot_convergence


def run_with_loss_tracking(attacker, prompt, target, method_name):
    """Run optimization and record per-step loss."""
    losses = attacker.optimize_with_logging(prompt, target)
    return losses


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Convergence Curves (Figure 3)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--behaviors", type=str, default="0,1,2,3,4",
                        help="Comma-separated behavior indices to evaluate")
    parser.add_argument("--output", type=str, default="./outputs/convergence",
                        help="Directory to save figure and data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum optimization steps")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    behavior_indices = [int(x.strip()) for x in args.behaviors.split(",")]
    all_behaviors = load_advbench()
    selected_behaviors = [all_behaviors[i] for i in behavior_indices if i < len(all_behaviors)]

    asa_attacker = ASAAttack(
        model_name=args.model,
        device=args.device,
        max_steps=args.max_steps,
    )
    gcg_attacker = GCGAttack(
        model_name=args.model,
        device=args.device,
        max_steps=args.max_steps,
    )

    convergence_data = {}

    for behavior in tqdm(selected_behaviors, desc="Running convergence experiments"):
        bid = behavior.get("id", behavior["prompt"][:20])
        print(f"\nBehavior: {bid}")

        asa_losses = run_with_loss_tracking(
            asa_attacker, behavior["prompt"], behavior["target"], "ASA"
        )
        gcg_losses = run_with_loss_tracking(
            gcg_attacker, behavior["prompt"], behavior["target"], "GCG"
        )

        convergence_data[bid] = {
            "asa": asa_losses,
            "gcg": gcg_losses,
        }

    # Save raw convergence data
    data_path = os.path.join(args.output, f"convergence_data_{timestamp}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(convergence_data, f, indent=2)
    print(f"Saved convergence data: {data_path}")

    # Plot convergence curves
    fig = plot_convergence(convergence_data)
    fig_path = os.path.join(args.output, f"convergence_curves_{timestamp}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
