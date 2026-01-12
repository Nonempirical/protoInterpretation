# ProtoInterpretation

A lightweight library for sampling **multiple stochastic continuations** from the same prompt and analyzing how the model’s “horizon” of possible continuations evolves over generation.

This project is **Colab-first**: it ships notebooks for generating runs (saved to Google Drive) and re-plotting report figures from those saved runs.

## What you can do with it

- **Sample chains**: generate N independent continuations (“chains”) for a prompt for T steps (tokens)
- **Measure horizon dynamics**: predictive entropy and embedding-space dispersion (“horizon width”)
- **Visualize structure**: PCA + UMAP projections, comparisons across prompts/runs
- **Save & reload runs**: store `batch.npz` + `metrics.json` per prompt in a Drive folder for reproducible re-plotting

## Installation

### Local (editable)

```bash
pip install -e .
```

### Google Colab (recommended)

```python
!git clone https://github.com/Nonempirical/protoInterpretation.git
%cd protoInterpretation
!pip install -e .
```

## Notebooks (how to reproduce the project data/figures)

The repo includes two notebooks under `src/protoInterpretation/`:

- **`protoInterpretation.ipynb`**: generate runs (sampling + metrics) and save to Drive
- **`reRunProtoInterpretation.ipynb`**: load saved runs from Drive and regenerate figures (entropy / horizon width / UMAP)

### Colab prerequisites (Hugging Face auth + Google Drive)

1) **Mount Drive**

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
```

2) **Authenticate to Hugging Face (for gated models like Llama)**

The notebooks expect a Colab Secret named **`hfKey`**:
Colab → **Runtime → Secrets** → add `hfKey` containing your Hugging Face token.

```python
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = userdata.get("hfKey")
login(token=HF_TOKEN)
```

3) **Runs folder**

Runs are saved under a directory named:

- **`protoInterpretation-runs`**

The rerun notebook searches common Drive locations (e.g., `MyDrive/` and `shared-with-me/`) using `find_runs_directory(...)`.

## Quick start (library API)

```python
from protoInterpretation import (
    ModelWrapper,
    SamplingConfig,
    sample_chain,
    compute_horizon_metrics,
    plot_entropy_curve,
    plot_horizon_width,
)

model = ModelWrapper.from_pretrained("gpt2")

cfg = SamplingConfig(
    num_chains=256,
    max_steps=32,
    temperature=0.9,
    top_k=0,
    top_p=0.9,
    seed=42,
    store_topk_logits=50,
    store_attention_weights=True,
)

batch = sample_chain(model, "The man", cfg)
metrics = compute_horizon_metrics(batch)

plot_entropy_curve(metrics)
plot_horizon_width(metrics)
```

## Generate runs and save to Drive (minimal workflow)

```python
from protoInterpretation import ModelWrapper, SamplingConfig, run_prompts_and_save

BASE_RUN_DIR = "/content/drive/MyDrive/protoInterpretation-runs"

model = ModelWrapper.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

cfg = SamplingConfig(
    num_chains=256,
    max_steps=32,
    temperature=0.9,
    top_k=0,
    top_p=0.9,
    seed=42,
    store_topk_logits=50,
    store_attention_weights=True,
)

prompts = [
    # Open
    "A bat is",
    "The woman in the blue dress",
    "I saw",
    "Something happens when there exists",
    "The man",
    "The man in the street",
    # Closed
    "A pencil is",
    "Napoleon is",
    "Photosynthesis is the process where",
    "The declaration of Independence formally",
    "Photosynthesis is",
    "Erosion is",
]

saved = run_prompts_and_save(model=model, prompts=prompts, cfg=cfg, base_run_dir=BASE_RUN_DIR)
print(f"Saved {len(saved)} runs into: {BASE_RUN_DIR}")
```

## Load and re-plot saved runs (minimal workflow)

```python
from protoInterpretation import find_runs_directory, scan_runs, load_run_by_name

BASE_RUN_DIR = find_runs_directory("/content/drive/MyDrive", folder_name="protoInterpretation-runs")
runs = scan_runs(BASE_RUN_DIR)

batch, metrics = load_run_by_name(next(iter(runs.keys())), runs, compute_metrics=True)
print(metrics.signature)
```

## Notes on what is measured (important for reporting)

- **Sampling vs measurement**: temperature and top‑p are applied to logits for *sampling tokens*, while entropy is computed from **pre-decoding** logits stored as top‑K (K=50) per step.
- **Entropy approximation**: entropy is computed by softmax-renormalizing over the stored top‑K logits; this can **underestimate** full-vocabulary entropy when probability mass lies outside the top‑K.
- **Step indexing**: steps are indexed `t = 0..31` for `max_steps=32`; each step corresponds to one generated token.

## Project structure

```
protoInterpretation/
├── README.md
├── requirements.txt
├── setup.py
└── src/
    └── protoInterpretation/
        ├── protoInterpretation.ipynb
        ├── reRunProtoInterpretation.ipynb
        ├── __init__.py
        ├── model.py
        ├── data_structures.py
        ├── sampling.py
        ├── analysis.py
        ├── projections.py
        ├── io.py
        ├── viz.py
        └── comparison.py
```

## License

MIT
