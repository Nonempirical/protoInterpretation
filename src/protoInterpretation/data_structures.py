from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


# -------------------------
# Basic config containers
# -------------------------

@dataclass
class SamplingConfig:
    """
    Configuration for how we sample full chains from a single prompt.
    """
    num_chains: int = 256          # number of independent generated chains
    max_steps: int = 32            # number of generated tokens per chain
    temperature: float = 1.0
    top_k: int = 0                 # 0 = disabled
    top_p: float = 1.0             # 1.0 = disabled
    seed: Optional[int] = None     # for reproducibility

    # How many logits to keep per step (for analysis/KL).
    store_topk_logits: int = 50
    
    # Store attention weights (weighted average across heads, last layer only)
    store_attention_weights: bool = False


@dataclass
class PromptSpec:
    """
    Description of a single prompt whose horizon we explore.
    """
    text: str
    # Optional user label (e.g. "open", "closed", "control") for grouping/comparison.
    label: Optional[str] = None
    # Optional arbitrary metadata (e.g. prompt family ID, notes).
    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Core sampled data
# -------------------------

@dataclass
class ChainBatch:
    """
    Raw outputs from sampling multiple full chains for a SINGLE prompt.
    This is the "ground truth" object that analysis/projections/viz should consume.

    Shapes:
      num_chains = N
      max_steps = T
      d_model   = D
      k_logits  = K
    """
    prompt: PromptSpec

    # Tokenized prompt (input context) as IDs.
    prompt_token_ids: np.ndarray          # shape: [prompt_len]

    # Generated token IDs for each chain and step.
    # If generation ends early, remaining positions can be padded with a special ID.
    token_ids: np.ndarray                 # shape: [N, T]

    # Last-token hidden state at each step for each chain.
    embeddings: np.ndarray                # shape: [N, T, D]

    # Top-k logits and corresponding token IDs for each step (optional but very useful).
    # If you don't want this, you can set these to None.
    topk_token_ids: Optional[np.ndarray] = None   # shape: [N, T, K]
    topk_logits: Optional[np.ndarray] = None      # shape: [N, T, K]

    # Optional masks / bookkeeping
    # 1 if this position is a "real" generated token (not padding), 0 otherwise.
    step_mask: Optional[np.ndarray] = None        # shape: [N, T]

    # Decoded text sequences (full prompt + generated tokens as strings)
    # One string per chain containing the complete text
    text_sequences: Optional[List[str]] = None     # length: N

    # Attention weights: weighted average (across heads) from last token to all tokens
    # Shape: [N, T, max_seq_len] where max_seq_len = prompt_len + max_steps
    # Each [n, t, :] contains attention pattern from token at step t to all previous tokens
    # Only stored if SamplingConfig.store_attention_weights=True
    attention_weights: Optional[np.ndarray] = None  # [N, T, max_seq_len]

    # Any extra info (e.g. RNG seeds, model name, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_chains(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.token_ids.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[-1])


# -------------------------
# Analysis outputs
# -------------------------

@dataclass
class EntropyCurve:
    """
    Entropy statistics over time for a ChainBatch.
    """
    # Mean entropy per step across chains.
    mean: np.ndarray          # shape: [T]
    # Optional: std dev, min, max over chains for each step.
    std: Optional[np.ndarray] = None      # shape: [T]
    per_chain: Optional[np.ndarray] = None  # shape: [N, T]


@dataclass
class HorizonWidthCurve:
    """
    Geometric width of the horizon over time.
    """
    # Max pairwise distance between chains at each step.
    max_dist: np.ndarray        # shape: [T]
    # Mean pairwise distance between chains at each step.
    mean_dist: np.ndarray       # shape: [T]
    # Optional: 95th percentile or other quantiles.
    p95_dist: Optional[np.ndarray] = None  # shape: [T]


@dataclass
class LineFitCurve:
    """
    How much a single line between two extremes explains the cloud at each step.
    """
    r2_per_step: np.ndarray      # shape: [T]
    # Optionally store indices of extreme pair used per step.
    extreme_pairs: Optional[np.ndarray] = None  # shape: [T, 2]


@dataclass
class KlDivergenceSummary:
    """
    KL divergence statistics between 'extreme' sequences (and/or overall pairs).
    """
    # Mean symmetric KL over time between chosen extreme pair.
    symmetric_kl_curve: Optional[np.ndarray] = None  # shape: [T]
    # Global scalar summaries (e.g. mean over steps).
    symmetric_kl_mean: Optional[float] = None


@dataclass
class ClusterSummary:
    """
    Clustering information in mid-dimensional space (e.g. PCA-50).
    """
    num_clusters: int
    labels: np.ndarray                      # shape: [N]
    # Optional: inertia, silhouette, etc.
    cluster_sizes: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpennessSignature:
    """
    Compact feature vector describing the "openness" of a prompt's horizon.
    This is what we'll compare across prompts.
    """
    # Scalars (or small arrays) you want to store; they can be standardized later.
    mean_entropy: float
    mean_horizon_width: float
    mean_linearity_r2: float
    mean_symmetric_kl: Optional[float] = None
    num_clusters: Optional[int] = None

    # Optionally: keep the raw curves for more detailed comparison.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HorizonMetrics:
    """
    Bundle all per-prompt analysis results in one object.
    """
    prompt: PromptSpec
    entropy: EntropyCurve
    width: HorizonWidthCurve
    line_fit: LineFitCurve
    kl: KlDivergenceSummary
    clusters: Optional[ClusterSummary] = None
    signature: Optional[OpennessSignature] = None

    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Projection outputs
# -------------------------

@dataclass
class ProjectionResult:
    """
    Dimensionality reduction for visualization.
    We keep both the mid-dimensional (PCA) and final (UMAP/other) projections.
    """
    # Original embeddings subset used for projection (for reference).
    # Usually shape: [N, D] where D is model or PCA dim.
    # Can be None if you don't want to store it.
    base_embeddings: Optional[np.ndarray]

    # PCA projection for analysis space (e.g. 50D).
    pca_embeddings: Optional[np.ndarray]       # shape: [N, D_pca]

    # 2D / 3D coordinates for visualization (e.g. UMAP).
    viz_embeddings_2d: Optional[np.ndarray] = None  # shape: [N, 2]
    viz_embeddings_3d: Optional[np.ndarray] = None  # shape: [N, 3]

    # Any diagnostic metrics (trustworthiness, distance correlation, etc.)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # What this projection represents (e.g. which step, which prompt).
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
