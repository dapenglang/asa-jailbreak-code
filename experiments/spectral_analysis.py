"""
spectral_analysis.py
Figure 4: Eigenvalue Spectrum

Run ASA on 4 models for 10 behaviors each, extract AFIM eigenvalues at convergence,
plot log-log eigenvalue spectrum with power-law fit, compute and display beta values,
and save figure to PNG.
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
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asa.asa_attack import ASAAttack
from data.advbench import load_advbench
from visualization.plots import plot_eigenvalue_spectrum


def extract_afim_eigenvalues(attacker, prompt, target):
    """Run optimization and extract top AFIM eigenvalues at convergence."""
    attacker.optimize(prompt, target)
    # AFIM is expected to be accessible after optimization
    afim = attacker.get_afim_matrix()
    # Compute eigenvalues (assumed symmetric positive semi-definite)
    eigenvalues = torch.linalg.eigvalsh(afim)
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    return eigenvalues.cpu().numpy()


def fit_power_law(eigenvalues):
    """Fit power-law to eigenvalue spectrum and return beta."""
    # Filter positive eigenvalues
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 2:
        return np.nan
    x = np.arange(1, len(ev) + 1)
    log_x = np.log(x)
    log_ev = np.log(ev)
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_ev)
    return slope, r_value ** 2


def main():
    parser = argparse.ArgumentParser(description="ASA Jailbreak: Eigenvalue Spectrum (Figure 4)")
    parser.add_argument("--models", type=str, required=True,
                        help="Comma-separated model names")
    parser.add_argument("--num_behaviors", type=int, default=10,
                        help="Number of behaviors per model")
    parser.add_argument("--output", type=str, default="./outputs/spectral",
                        help="Directory to save figure and data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_names = [m.strip() for m in args.models.split(",")]
    all_behaviors = load_advbench()

    spectral_data = {}
    beta_records = []

    for model_name in model_names:
        print(f"\n=== Model: {model_name} ===")
        attacker = ASAAttack(
            model_name=model_name,
            device=args.device,
        )

        behaviors = all_behaviors[:args.num_behaviors]
        all_eigenvalues = []

        for behavior in tqdm(behaviors, desc=f"Extracting eigenvalues [{model_name}]"):
            ev = extract_afim_eigenvalues(attacker, behavior["prompt"], behavior["target"])
            all_eigenvalues.append(ev)

        # Average eigenvalues across behaviors
        min_len = min(len(ev) for ev in all_eigenvalues)
        trimmed = [ev[:min_len] for ev in all_eigenvalues]
        mean_eigenvalues = np.mean(trimmed, axis=0)

        slope, r2 = fit_power_law(mean_eigenvalues)
        beta = -slope  # beta is the negative slope in log-log space

        spectral_data[model_name] = {
            "eigenvalues": mean_eigenvalues.tolist(),
            "beta": float(beta),
            "r2": float(r2),
        }

        beta_records.append({
            "model": model_name,
            "beta": round(beta, 4),
            "r2": round(r2, 4),
        })

        print(f"Model: {model_name} | Beta: {beta:.4f} | R^2: {r2:.4f}")

    # Save spectral data
    data_path = os.path.join(args.output, f"spectral_data_{timestamp}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(spectral_data, f, indent=2)
    print(f"\nSaved spectral data: {data_path}")

    # Save beta table
    beta_path = os.path.join(args.output, f"beta_values_{timestamp}.json")
    with open(beta_path, "w", encoding="utf-8") as f:
        json.dump(beta_records, f, indent=2)
    print(f"Saved beta values: {beta_path}")

    # Plot eigenvalue spectrum
    fig = plot_eigenvalue_spectrum(spectral_data)
    fig_path = os.path.join(args.output, f"eigenvalue_spectrum_{timestamp}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
