# Nextword Horizon

A lightweight library for sampling LLM outputs to analyze how the horizon of possible output changes with different prompts.

## Features

- **Colab-first**: Designed for easy use in Google Colab notebooks
- **Minimal dependencies**: Just torch, transformers, umap-learn, matplotlib, numpy, scikit-learn
- **Simple API**: Import and use - no heavy CLI or config system
- **Horizon analysis**: Compute metrics like entropy, horizon width, linearity, and KL divergence
- **Visualizations**: Plot projections, metrics, and comparisons

## Installation

### Local Installation

```bash
pip install -r requirements.txt
# Or install as a package:
pip install -e .
```

### Colab Installation

In Google Colab, clone the repository and install:

```python
# Clone the repository
!git clone https://github.com/yourusername/nextword-horizon-v2.git
%cd nextword-horizon-v2

# Install dependencies
!pip install torch transformers umap-learn matplotlib numpy scikit-learn

# Install the package (optional - you can also just add to path)
!pip install -e .
```

Or if you prefer to use it without installation:

```python
# Clone and add to path
!git clone https://github.com/yourusername/nextword-horizon-v2.git
import sys
sys.path.insert(0, '/content/nextword-horizon-v2')

# Install dependencies
!pip install torch transformers umap-learn matplotlib numpy scikit-learn
```

## Quick Start

```python
from src.nextword_horizon import (
    ModelWrapper, 
    SamplingConfig, 
    sample_chain, 
    compute_horizon_metrics,
    plot_entropy_curve,
    plot_horizon_width,
    project_step_embeddings,
    plot_step_scatter_2d
)

# Load model (supports local paths or HuggingFace hub names)
model = ModelWrapper.from_pretrained("gpt2")  # or "/path/to/local/model"

# Configure sampling
cfg = SamplingConfig(
    num_chains=128,
    max_steps=32,
    temperature=1.0,
    top_k=40,
    top_p=0.9,
    store_topk_logits=50,
    seed=42
)

# Sample outputs for a prompt
batch = sample_chain(model, "The future of AI is", cfg)

# Compute horizon metrics
metrics = compute_horizon_metrics(batch)
print(f"Mean entropy: {metrics.signature.mean_entropy:.2f} bits")
print(f"Mean horizon width: {metrics.signature.mean_horizon_width:.4f}")

# Visualize metrics
plot_entropy_curve(metrics)
plot_horizon_width(metrics)
```

## Compare Different Prompts

```python
# Sample with different prompts
batch1 = sample_chain(model, "The future of AI is", cfg)
batch2 = sample_chain(model, "The future of AI will be", cfg)

# Compute metrics for each
metrics1 = compute_horizon_metrics(batch1)
metrics2 = compute_horizon_metrics(batch2)

# Compare signatures
print(f"Entropy difference: {metrics2.signature.mean_entropy - metrics1.signature.mean_entropy:.2f}")
print(f"Width difference: {metrics2.signature.mean_horizon_width - metrics1.signature.mean_horizon_width:.4f}")
```

## Dimensionality Reduction and Visualization

```python
from src.nextword_horizon import project_step_embeddings, plot_step_scatter_2d

# Project embeddings at a specific step
proj = project_step_embeddings(batch, step=10)

# Plot 2D scatter with cluster labels
plot_step_scatter_2d(proj, labels=metrics.clusters.labels)

# Access diagnostics
print(f"Trustworthiness: {proj.diagnostics.get('trustworthiness', 'N/A')}")
print(f"Distance correlation: {proj.diagnostics.get('distance_correlation', 'N/A')}")
```

## Using Local Models in Colab

```python
# Download model to temp directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile

temp_dir = tempfile.mkdtemp()
model_name = "gpt2"

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(temp_dir)
tokenizer.save_pretrained(temp_dir)

# Use with Nextword Horizon
from src.nextword_horizon import ModelWrapper
wrapper = ModelWrapper.from_pretrained(temp_dir)
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
