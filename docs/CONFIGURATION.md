# Configuration Guide

This document explains all configuration files and hyperparameters.

## Files

| File | Purpose |
|------|---------|
| `configs/models.yaml` | Supported LLM definitions and chat templates |
| `configs/asa_default.yaml` | Default ASA hyperparameters from the paper |

## Model Configuration (`configs/models.yaml`)

### Structure

```yaml
models:
  <model_key>:
    name: <huggingface_model_id>
    dtype: <torch_dtype>
    device_map: <device_strategy>
    max_length: <max_token_length>
    template: <template_key>

templates:
  <template_key>: <prompt_format_string>
```

### Fields

- `name`: Hugging Face model identifier (e.g., `lmsys/vicuna-7b-v1.5`)
- `dtype`: Model precision. Options: `float16` (default), `bfloat16`, `float32`
- `device_map`: Device placement. Options: `auto` (recommended), `cuda:0`, `cpu`
- `max_length`: Maximum sequence length for tokenizer
- `template`: Reference to chat template in `templates` section

## Attack Configuration (`configs/asa_default.yaml`)

### Attack Parameters

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| `suffix_length` | 20 | Number of adversarial suffix tokens | Section 4.3 |
| `num_steps` | 500 | Maximum optimization steps | Section 5.1 |
| `batch_size` | 512 | Candidate batch size for evaluation | Algorithm 1 |
| `topk` | 256 | Top-k token candidates per position | Section 4.3 |

### Gumbel-Softmax Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature.init` | 2.0 | Initial Gumbel-Softmax temperature |
| `temperature.min` | 0.5 | Minimum temperature (floor) |
| `temperature.gamma` | 0.95 | Annealing rate per step |

### AFIM Parameters

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| `afim.window_size` | 50 | Gradient history window for covariance | Section 3.2 |
| `afim.update_frequency` | 1 | Update AFIM every N steps | Algorithm 1 |

### Subspace Parameters

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| `subspace.dim` | 32 | Active subspace dimension k | Section 4.2 |
| `subspace.update_frequency` | 10 | Update subspace every N steps | Algorithm 1 |
| `subspace.min_steps_before_subspace` | 20 | Gradient collection before first SVD | Section 5.1 |

### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization.early_stop_threshold` | 0.01 | Stop if loss < threshold |
| `optimization.early_stop_patience` | 20 | Steps without improvement |

## Loading Custom Configurations

### Method 1: YAML File

```python
import yaml
from asa.asa_attack import ASAAttack

with open("configs/asa_default.yaml") as f:
    config = yaml.safe_load(f)

# Override specific values
config["attack"]["subspace"]["dim"] = 64
config["attack"]["num_steps"] = 1000

attacker = ASAAttack(model, tokenizer, config=config)
```

### Method 2: Dictionary Directly

```python
config = {
    "batch_size": 256,
    "subspace_dim": 64,
    "num_steps": 1000,
    "suffix_length": 30
}
attacker = ASAAttack(model, tokenizer, config=config)
```

### Method 3: Command Line (Experiment Scripts)

All experiment scripts accept config overrides:

```bash
python experiments/main_results.py \
    --model vicuna-7b \
    --config configs/asa_default.yaml \
    --subspace_dim 64 \
    --num_steps 1000
```

## Recommended Configurations

### Fast Testing (Colab Free T4)

```python
fast_config = {
    "num_steps": 100,
    "batch_size": 128,
    "suffix_length": 10,
    "subspace_dim": 16,
    "topk": 128
}
```

### Paper Reproduction (A100)

Use `configs/asa_default.yaml` as-is. Expected runtime: ~5-10 min per behavior.

### Transfer Attack (Higher Quality)

```python
transfer_config = {
    "num_steps": 1000,
    "suffix_length": 30,
    "subspace_dim": 64,
    "afim_window": 100
}
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache directory |
| `CUDA_VISIBLE_DEVICES` | All | GPU selection |
| `PYTORCH_CUDA_ALLOC_CONF` | - | CUDA memory configuration |
