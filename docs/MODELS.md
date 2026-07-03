# Model Weights and Configuration Guide

This document details all supported models, their Hugging Face identifiers, weight download instructions, and code integration.

## Supported Models

All models are loaded via `transformers.AutoModelForCausalLM.from_pretrained()` with the identifiers below. Weights are automatically downloaded from Hugging Face on first use.

### Open-Source Chat Models

| Model | Hugging Face ID | Parameters | VRAM (fp16) | License | Chat Template |
|-------|----------------|------------|-------------|---------|---------------|
| **Vicuna-7B** | `lmsys/vicuna-7b-v1.5` | 7B | ~14 GB | Llama 2 | `USER: {prompt}\nASSISTANT:` |
| **Vicuna-13B** | `lmsys/vicuna-13b-v1.5` | 13B | ~26 GB | Llama 2 | `USER: {prompt}\nASSISTANT:` |
| **LLaMA-2-7B-Chat** | `meta-llama/Llama-2-7b-chat-hf` | 7B | ~14 GB | Llama 2 | `[INST] {prompt} [/INST]` |
| **LLaMA-2-13B-Chat** | `meta-llama/Llama-2-13b-chat-hf` | 13B | ~26 GB | Llama 2 | `[INST] {prompt} [/INST]` |
| **LLaMA-3-8B-Instruct** | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | Llama 3 | `<|begin_of_text|>...` |
| **Mistral-7B-Instruct** | `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ~14 GB | Apache 2.0 | `[INST] {prompt} [/INST]` |
| **Gemma-7B-it** | `google/gemma-7b-it` | 7B | ~14 GB | Gemma | `<start_of_turn>user\n{prompt}...` |
| **Phi-4** | `microsoft/phi-4` | 7B | ~14 GB | MIT | `<|user|>\n{prompt}...` |

### Code Integration

The code automatically selects the correct chat template from `configs/models.yaml`:

```python
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model config
with open("configs/models.yaml") as f:
    model_config = yaml.safe_load(f)["models"]["vicuna-7b"]

# Download and load (weights auto-downloaded)
model = AutoModelForCausalLM.from_pretrained(
    model_config["name"],           # "lmsys/vicuna-7b-v1.5"
    torch_dtype=getattr(torch, model_config["dtype"]),  # torch.float16
    device_map=model_config["device_map"],               # "auto"
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
```

### Gated Models (Require Authentication)

The following models require Hugging Face authentication and license acceptance:

- **LLaMA-2/3 family** (`meta-llama/*`): Request access at https://huggingface.co/meta-llama
- **Gemma** (`google/gemma-7b-it`): Request access at https://huggingface.co/google/gemma-7b-it

Steps:
1. Create Hugging Face account
2. Get access token: https://huggingface.co/settings/tokens
3. Accept model license on the model page
4. Login before loading:

```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

### Weight Caching

By default, weights are cached in `~/.cache/huggingface/hub/` (~14-16GB per 7B model). To use a custom cache:

```python
import os
os.environ["HF_HOME"] = "/path/to/large/disk/hf_cache"
```

On Google Colab, mount Drive and set cache there to persist across sessions.

### Local Weight Path (Offline Mode)

If you pre-download weights, specify local path:

```python
# If weights saved to ./models/vicuna-7b-v1.5/
model = AutoModelForCausalLM.from_pretrained(
    "./models/vicuna-7b-v1.5",
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
```

## Adding New Models

To add a new model, edit `configs/models.yaml`:

```yaml
models:
  your-model-name:
    name: "organization/model-id"
    dtype: "float16"      # or "bfloat16"
    device_map: "auto"
    max_length: 512
    template: "your_template"

templates:
  your_template: "User: {prompt}\nAssistant:"
```

Then use in code:

```python
attacker = ASAAttack(model, tokenizer)
result = attacker.attack(prompt, target)
```
