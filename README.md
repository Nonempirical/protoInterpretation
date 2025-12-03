# Nextword Horizon

A lightweight library for sampling LLM outputs to analyze how the horizon of possible output changes with different prompts.

## Features

- **Colab-first**: Designed for easy use in Google Colab notebooks
- **Minimal dependencies**: Just torch, transformers, umap-learn, matplotlib, numpy
- **Simple API**: Import and use - no heavy CLI or config system
- **Horizon analysis**: Compute metrics like entropy, diversity, and top-k coverage
- **Visualizations**: Plot projections, metrics, and comparisons

## Installation

```bash
pip install -r requirements.txt
```

Or install in Colab:
```python
!pip install torch transformers umap-learn matplotlib numpy scikit-learn
```

## Quick Start

```python
from src.nextword_horizon import ModelWrapper, sample_chain, compute_horizon_metrics, plot_horizon_metrics

# Load model (supports local paths or HuggingFace hub names)
model = ModelWrapper("gpt2")  # or "/path/to/local/model"

# Sample outputs for a prompt
run1 = sample_chain(
    model=model,
    prompt="The future of AI is",
    max_length=30,
    num_samples=100,
    temperature=1.0
)

# Compute horizon metrics
metrics = compute_horizon_metrics(run1)
print(f"Entropy: {metrics.entropy:.2f} bits")
print(f"Unique tokens: {metrics.unique_tokens}")
print(f"Unique sequences: {metrics.unique_sequences}")

# Visualize metrics
plot_horizon_metrics(metrics)
```

## Compare Different Prompts

```python
# Sample with different prompts
run1 = sample_chain(model, "The future of AI is", num_samples=100)
run2 = sample_chain(model, "The future of AI will be", num_samples=100)

# Compare horizons
from src.nextword_horizon import compare_horizons, plot_comparison

comparison = compare_horizons(run1, run2)
plot_comparison(run1, run2, label1="is", label2="will be")
```

## Dimensionality Reduction

```python
from src.nextword_horizon import project_logits_pca, project_logits_umap, plot_projection

# Project logits to 2D using PCA
pca_result = project_logits_pca(run1, n_components=2)
plot_projection(pca_result, title="PCA Projection")

# Or use UMAP
umap_result = project_logits_umap(run1, n_components=2)
plot_projection(umap_result, title="UMAP Projection")
```

## Using Local Models in Colab

```python
# Download model to temp directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile
import os

temp_dir = tempfile.mkdtemp()
model_name = "gpt2"

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(temp_dir)
tokenizer.save_pretrained(temp_dir)

# Use with Nextword Horizon
from src.nextword_horizon import ModelWrapper
wrapper = ModelWrapper(temp_dir)
```

## Project Structure

```
nextword-horizon-v2/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── nextword_horizon/
        ├── __init__.py
        ├── model.py           # HF model wrapper
        ├── data_structures.py # Core dataclasses
        ├── sampling.py        # Full-chain sampling
        ├── analysis.py        # Openness metrics
        ├── projections.py     # PCA/UMAP
        └── viz.py             # Plotting helpers
```

## License

MIT

