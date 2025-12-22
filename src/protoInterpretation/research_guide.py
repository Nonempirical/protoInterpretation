from __future__ import annotations

from typing import Optional

try:
    from IPython.display import display, Markdown, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Fallback for non-notebook environments
    def display(x):
        print(x)
    def Markdown(x):
        return x
    def HTML(x):
        return x


def print_research_guide(
    runs: Optional[dict] = None,
    show_examples: bool = True,
    output_format: str = "markdown",
) -> None:
    """
    Print a comprehensive research guide showing available data and functions.
    
    Args:
        runs: Optional dictionary of runs (from scan_runs) to show available datasets
        show_examples: If True, include code examples
        output_format: "markdown" or "html" for display format
    """
    guide = _build_research_guide(runs, show_examples)
    
    if output_format == "html":
        display(HTML(_markdown_to_html(guide)))
    else:
        display(Markdown(guide))


def _build_research_guide(runs: Optional[dict], show_examples: bool) -> str:
    """Build the research guide markdown content."""
    
    guide = """# ProtoInterpretation Research Guide

## 📊 Available Data Structures

### ChainBatch
The core data structure containing sampled chains:
- `prompt`: PromptSpec with text and label
- `token_ids`: [N, T] Generated token IDs
- `embeddings`: [N, T, D] Last-token hidden states per step
- `topk_token_ids`: [N, T, K] Top-k token IDs (optional)
- `topk_logits`: [N, T, K] Top-k logits (optional)
- `step_mask`: [N, T] Mask for valid steps
- `meta`: Dictionary with metadata (model, config, timestamp, etc.)

### HorizonMetrics
Computed metrics for a ChainBatch:
- `entropy`: EntropyCurve (mean, std, per_chain)
- `width`: HorizonWidthCurve (max_dist, mean_dist, p95_dist)
- `line_fit`: LineFitCurve (r2_per_step, extreme_pairs)
- `kl`: KlDivergenceSummary (symmetric_kl_curve, symmetric_kl_mean)
- `clusters`: ClusterSummary (labels, cluster_sizes)
- `signature`: OpennessSignature (summary scalars)

### ProjectionResult
Dimensionality reduction results:
- `base_embeddings`: Original embeddings [N, D]
- `pca_embeddings`: PCA projection [N, D_pca]
- `viz_embeddings_2d`: UMAP 2D projection [N, 2]
- `diagnostics`: Quality metrics (trustworthiness, distance_correlation)

---

## 🔧 Data Loading Functions

### Finding and Scanning Runs
```python
from protoInterpretation import find_runs_directory, scan_runs

# Find runs directory in Google Drive
BASE_RUN_DIR = find_runs_directory("/content/drive/MyDrive", 
                                   folder_name="protoInterpretation-runs")

# Scan for available runs
runs = scan_runs(BASE_RUN_DIR)  # Returns: {run_name: run_path}
```

"""
    
    if runs:
        guide += f"\n### Available Runs ({len(runs)} total)\n\n"
        for i, (name, path) in enumerate(sorted(runs.items()), 1):
            guide += f"{i}. `{name}`\n"
        guide += "\n"
    
    guide += """### Loading Data
```python
from protoInterpretation import load_run_by_name, load_embeddings_from_runs

# Load a single run (with metrics)
batch, metrics = load_run_by_name("run_name", runs)

# Load embeddings from multiple runs (for animation)
emb_per_run, min_T, D = load_embeddings_from_runs(["run1", "run2"], runs)
```

---

## 📈 Analysis Functions

### Computing Metrics
```python
from protoInterpretation import compute_horizon_metrics

# Compute all metrics for a batch
metrics = compute_horizon_metrics(batch)

# Access specific metrics
mean_entropy = metrics.entropy.mean
horizon_width = metrics.width.mean_dist
linearity = metrics.line_fit.r2_per_step
kl_divergence = metrics.kl.symmetric_kl_mean
```

### Dimensionality Reduction
```python
from protoInterpretation import project_step_embeddings, compute_global_umap

# Project embeddings at a specific step
proj = project_step_embeddings(batch, step=10)

# Compute global UMAP over multiple runs and all steps
embeddings_2d, run_labels, step_labels, chain_indices = compute_global_umap(emb_per_run)
```

---

## 🎨 Visualization Functions

### Time-Series Plots
```python
from protoInterpretation import (
    plot_entropy_curve,
    plot_horizon_width,
    plot_entropy_comparison,
    plot_horizon_width_comparison,
)

# Plot entropy over time
plot_entropy_curve(metrics)

# Plot horizon width over time (Mean / P95 / Max)
plot_horizon_width(metrics)

# Compare multiple runs on one axis
plot_entropy_comparison(metrics_per_run, labels=display_names)
plot_horizon_width_comparison(metrics_per_run, labels=display_names)
```

### 2D Scatter Plots
```python
from protoInterpretation import plot_step_scatter_2d

# Plot 2D projection at a step
plot_step_scatter_2d(proj, labels=metrics.clusters.labels)
```

### Global UMAP Facets (Matplotlib)
```python
from protoInterpretation import compute_global_umap, plot_global_umap_facets_by_step

E_2d, run_labels, step_labels, chain_indices = compute_global_umap(emb_per_run)
plot_global_umap_facets_by_step(E_2d, run_labels, step_labels, labels=display_names)
```

### Final-step UMAP across runs (Matplotlib)
```python
from protoInterpretation import plot_final_step_umap_across_runs

fig, ax, E_2d, run_labels, chain_indices = plot_final_step_umap_across_runs(
    emb_per_run,
    labels=display_names,
    step=-1,
)
```

### Comparison Plots
```python
from protoInterpretation import (
    plot_metrics_comparison,
    plot_projections_comparison,
    plot_joint_umap,
    compare_runs,
)

# Compare two runs side-by-side
compare_runs(batch_A, batch_B, metrics_A, metrics_B, step=10)

# Or create individual comparison plots
plot_metrics_comparison(metrics_A, metrics_B, label_A="Run A", label_B="Run B")
plot_projections_comparison(proj_A, proj_B)
plot_joint_umap(embeddings_A, embeddings_B)
```

### Animated Visualizations
```python
from protoInterpretation import plot_animated_umap

# Create animated UMAP visualization
fig = plot_animated_umap(
    embeddings_2d=embeddings_2d,
    run_labels=run_labels,
    step_labels=step_labels,
    title="Global UMAP horizon movie",
    animation_frame_duration=100,  # Speed in milliseconds
)
fig.show()
```

---

## 🔬 Common Research Workflows

### 1. Compare Two Prompts
```python
# Load two runs
batch_A, metrics_A = load_run_by_name("prompt_A", runs)
batch_B, metrics_B = load_run_by_name("prompt_B", runs)

# Compare comprehensively
results = compare_runs(batch_A, batch_B, metrics_A, metrics_B)

# Access comparison results
print(f"Entropy difference: {metrics_B.signature.mean_entropy - metrics_A.signature.mean_entropy}")
print(f"Width difference: {metrics_B.signature.mean_horizon_width - metrics_A.signature.mean_horizon_width}")
```

### 2. Analyze Single Prompt Evolution
```python
# Load run
batch, metrics = load_run_by_name("my_run", runs)

# Plot evolution over time
plot_entropy_curve(metrics)
plot_horizon_width(metrics)

# Visualize at different steps
for step in [0, 10, 20, -1]:
    proj = project_step_embeddings(batch, step=step)
    plot_step_scatter_2d(proj, title=f"Step {step}")
```

### 3. Animate Multiple Runs Together
```python
# Load embeddings from multiple runs
emb_per_run, min_T, D = load_embeddings_from_runs(["run1", "run2", "run3"], runs)

# Compute global UMAP
embeddings_2d, run_labels, step_labels, chain_indices = compute_global_umap(emb_per_run)

# Create animation
fig = plot_animated_umap(
    embeddings_2d=embeddings_2d,
    run_labels=run_labels,
    step_labels=step_labels,
    animation_frame_duration=100,
)
fig.show()
```

### 4. Cluster Analysis
```python
# Metrics include clustering by default
metrics = compute_horizon_metrics(batch)

# Access cluster labels
cluster_labels = metrics.clusters.labels
num_clusters = metrics.clusters.num_clusters
cluster_sizes = metrics.clusters.cluster_sizes

# Visualize with cluster colors
proj = project_step_embeddings(batch, step=-1)
plot_step_scatter_2d(proj, labels=cluster_labels)
```

### 5. Extract Openness Signature
```python
# Get compact feature vector
signature = metrics.signature

print(f"Mean Entropy: {signature.mean_entropy:.2f} bits")
print(f"Mean Horizon Width: {signature.mean_horizon_width:.4f}")
print(f"Mean Linearity R²: {signature.mean_linearity_r2:.4f}")
print(f"Mean Symmetric KL: {signature.mean_symmetric_kl:.4f}")
print(f"Number of Clusters: {signature.num_clusters}")
```

---

## 💾 Saving Data

```python
from protoInterpretation import (
    save_batch_npz,
    save_metrics_json,
    run_prompts_and_save,
    save_horizon_run_from_prompt,
)

# Save batch (numeric data only)
save_batch_npz(batch, "path/to/batch.npz")

# Save metrics (with metadata)
save_metrics_json(metrics, "path/to/metrics.json", batch_meta=batch.meta)

# Convenience: sample -> compute -> save per prompt
saved = run_prompts_and_save(model, prompts, cfg, base_run_dir="path/to/runs")
```

---

## 📚 Key Metrics Explained

- **Entropy**: Uncertainty in token predictions (higher = more diverse outputs)
- **Horizon Width**: Geometric spread of embeddings (higher = more divergent paths)
- **Linearity R²**: How well embeddings fit a 1D line (lower = more complex structure)
- **Symmetric KL**: Divergence between extreme sequences (higher = more different)
- **Clusters**: Number of distinct groups in embedding space

---

## 🎯 Quick Reference

**Load**: `find_runs_directory`, `scan_runs`, `load_run_by_name`, `load_embeddings_from_runs`

**Analyze**: `compute_horizon_metrics`, `project_step_embeddings`, `compute_global_umap`

**Visualize**: `plot_entropy_curve`, `plot_horizon_width`, `plot_entropy_comparison`, `plot_horizon_width_comparison`, `plot_final_step_umap_across_runs`, `plot_global_umap_facets_by_step`, `plot_step_scatter_2d`, `plot_metrics_comparison`, `plot_projections_comparison`, `plot_joint_umap`, `plot_animated_umap`, `compare_runs`

**Save**: `save_batch_npz`, `save_metrics_json`, `save_horizon_run`, `save_horizon_run_from_prompt`, `run_prompts_and_save`

---

*For more details, see the function docstrings or the project README.*
"""
    
    return guide


def _markdown_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML (simple version)."""
    import re
    
    html = markdown_text
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Code blocks
    html = re.sub(r'```python\n(.*?)```', r'<pre><code class="language-python">\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Line breaks
    html = html.replace('\n', '<br>\n')
    
    return f'<div style="font-family: Arial, sans-serif; padding: 20px;">{html}</div>'

