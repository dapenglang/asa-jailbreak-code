# Experiment Reproduction Guide

This document provides detailed instructions for reproducing all experiments from the paper.

## Hardware Requirements

| Experiment | GPU | VRAM | Time |
|------------|-----|------|------|
| Single attack | A100 | 14GB | ~5 min |
| Table 3 (50 behaviors, 4 models) | A100 | 14GB | ~8 hours |
| Figure 4 (spectral analysis) | A100 | 14GB | ~2 hours |
| All figures + tables | A100 | 14GB | ~24 hours |

## Datasets

### AdvBench

Automatically loaded via Hugging Face `datasets`:

```python
from data.advbench import load_advbench
behaviors = load_advbench(split="harmful", num_samples=50)
```

Source: `walledai/AdvBench` on Hugging Face.

### HarmBench

```python
from data.harmbench import load_harmbench
behaviors = load_harmbench(source="hf")
```

Source: `cais/harmbench` on Hugging Face. Judge evaluation requires additional setup (see README).

## Main Results

### Table 3: AdvBench Main Results

```bash
python experiments/main_results.py \
    --model vicuna-7b,llama2-7b-chat,mistral-7b-instruct,gemma-7b-it \
    --method asa,gcg \
    --dataset advbench \
    --num_behaviors 50 \
    --output_dir ./results/table3
```

Output: `./results/table3/results.json` and `summary.csv`

### Table 4: HarmBench Main Results

```bash
python experiments/main_results.py \
    --model vicuna-7b,llama2-7b-chat,mistral-7b-instruct,gemma-7b-it \
    --method asa,gcg \
    --dataset harmbench \
    --num_behaviors 50 \
    --output_dir ./results/table4
```

## Figures

### Figure 3: Convergence Curves

```bash
python experiments/convergence.py \
    --model llama2-7b-chat \
    --behaviors 0,1,2,3,4 \
    --output ./results/figures/convergence.png
```

Plots loss vs. optimization step for ASA and GCG.

### Figure 4: Eigenvalue Spectrum

```bash
python experiments/spectral_analysis.py \
    --models vicuna-7b,llama2-7b-chat,mistral-7b-instruct,llama3-8b-instruct \
    --num_behaviors 10 \
    --output ./results/figures/eigenvalue_spectrum.png
```

Plots eigenvalue decay and power-law fit for each model.

### Figure 5: Dimensionality Ablation

```bash
python experiments/dimensionality.py \
    --model llama2-7b-chat \
    --ks 4,8,16,24,32,40,48,64,80,128 \
    --num_behaviors 20 \
    --output ./results/figures/dimensionality.png
```

Plots ASR vs. subspace dimension k.

### Figure 6: Turbulence Profile

```bash
python experiments/turbulence.py \
    --model llama2-7b-chat \
    --num_behaviors 10 \
    --output ./results/figures/turbulence.png
```

Plots layer-wise gradient norm (turbulence).

## Ablation Study

### Table 6: Component Ablation

```bash
python experiments/ablation.py \
    --model llama2-7b-chat \
    --num_behaviors 20 \
    --output_dir ./results/ablation
```

Tests 9 ASA variants (full, no subspace, no Gumbel, etc.).

## Defense Evaluation

### Table 8: Defense Evasion

```bash
python experiments/defense_evasion.py \
    --model llama2-7b-chat \
    --methods asa,gcg \
    --num_behaviors 50 \
    --output_dir ./results/defense
```

Evaluates against perplexity filter, SmoothLLM, and SafeDecoding.

## Output Structure

```
results/
├── table3/
│   ├── results.json
│   └── summary.csv
├── table4/
│   └── ...
├── figures/
│   ├── convergence.png
│   ├── eigenvalue_spectrum.png
│   ├── dimensionality.png
│   └── turbulence.png
├── ablation/
│   └── results.json
└── defense/
    └── results.json
```

## Reproducing Paper Numbers

Expected results (approximate, may vary ±2-3% due to randomness):

| Model | ASA ASR | GCG ASR |
|-------|---------|---------|
| Vicuna-7B | 96.0% | 88.0% |
| LLaMA-2-7B | 91.0% | 82.0% |
| Mistral-7B | 94.0% | 85.0% |
| Gemma-7B | 89.0% | 79.0% |

## Notes

- Results may vary slightly across runs due to Gumbel noise
- For exact reproducibility, set `torch.manual_seed(42)` before each run
- On V100 (16GB), use `batch_size=256` to avoid OOM
- On A100 (40GB), use `batch_size=512` (default) for best performance
