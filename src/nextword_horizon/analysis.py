from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

from .data_structures import (
    ChainBatch,
    EntropyCurve,
    HorizonWidthCurve,
    LineFitCurve,
    KlDivergenceSummary,
    ClusterSummary,
    OpennessSignature,
    HorizonMetrics,
)


# -------------------------
# Entropy over time
# -------------------------

def _softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in numpy."""
    x = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _entropy_from_logits_topk(topk_logits: np.ndarray) -> np.ndarray:
    """
    Approximate entropy from top-k logits.

    topk_logits: [N, T, K]
    Returns:
        entropies: [N, T] in bits.
    Note: This approximates the true entropy because we only
    consider the stored top-k tokens and renormalize.
    """
    probs = _softmax_np(topk_logits, axis=-1)  # [N, T, K]
    # avoid log(0)
    eps = 1e-12
    log_probs = np.log2(probs + eps)
    ent = -np.sum(probs * log_probs, axis=-1)  # [N, T]
    return ent


def compute_entropy_curve(batch: ChainBatch) -> EntropyCurve:
    """
    Compute entropy statistics over time from the stored top-k logits.

    Returns EntropyCurve with:
      - mean: [T] mean entropy across chains
      - std:  [T] std across chains
      - per_chain: [N, T] individual entropies
    """
    if batch.topk_logits is None:
        raise ValueError(
            "Cannot compute entropy: topk_logits is None. "
            "Set SamplingConfig.store_topk_logits > 0 when sampling."
        )

    per_chain = _entropy_from_logits_topk(batch.topk_logits)  # [N, T]

    # If a step_mask exists, you could use it here to ignore padded steps.
    # For now, everything is "real".
    mean = per_chain.mean(axis=0)  # [T]
    std = per_chain.std(axis=0)    # [T]

    return EntropyCurve(
        mean=mean,
        std=std,
        per_chain=per_chain,
    )


# -------------------------
# Horizon width over time
# -------------------------

def _pairwise_distances_for_step(
    embeddings_step: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise distances between chains at a single step.

    embeddings_step: [N, D]
    Returns:
        dist: [N, N]
    """
    return pairwise_distances(embeddings_step, metric=metric)


def compute_horizon_width_curve(
    batch: ChainBatch,
    metric: str = "cosine",
) -> HorizonWidthCurve:
    """
    Compute geometric "width" of the horizon over time.

    For each step t:
      - max_dist[t]: max pairwise distance across chains
      - mean_dist[t]: mean pairwise distance (upper triangle)
      - p95_dist[t]: 95th percentile of pairwise distances
    """
    embeddings = batch.embeddings  # [N, T, D]
    N, T, _ = embeddings.shape

    max_dist = np.zeros(T, dtype=np.float32)
    mean_dist = np.zeros(T, dtype=np.float32)
    p95_dist = np.zeros(T, dtype=np.float32)

    for t in range(T):
        emb_t = embeddings[:, t, :]  # [N, D]
        dist_mat = _pairwise_distances_for_step(emb_t, metric=metric)  # [N, N]

        # Use upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(N, k=1)
        vals = dist_mat[triu_indices]
        if vals.size == 0:
            # degenerate case: N=1
            max_dist[t] = 0.0
            mean_dist[t] = 0.0
            p95_dist[t] = 0.0
        else:
            max_dist[t] = float(vals.max())
            mean_dist[t] = float(vals.mean())
            p95_dist[t] = float(np.percentile(vals, 95.0))

    return HorizonWidthCurve(
        max_dist=max_dist,
        mean_dist=mean_dist,
        p95_dist=p95_dist,
    )


# -------------------------
# Line fit curve (1D vs multi-D)
# -------------------------

def _extreme_pair_indices(
    dist_mat: np.ndarray,
) -> Tuple[int, int]:
    """
    Given a pairwise distance matrix [N, N], find indices (i, j) of the
    most distant pair (upper triangle).
    """
    N = dist_mat.shape[0]
    triu_indices = np.triu_indices(N, k=1)
    vals = dist_mat[triu_indices]
    if vals.size == 0:
        return 0, 0
    max_idx = int(np.argmax(vals))
    i = int(triu_indices[0][max_idx])
    j = int(triu_indices[1][max_idx])
    return i, j


def _line_fit_r2_for_step(
    embeddings_step: np.ndarray,
    i: int,
    j: int,
) -> float:
    """
    Given embeddings [N, D] and an extreme pair (i, j),
    compute how much variance is explained by the line between them.

    R^2 = var(proj_scores) / total_variance
    """
    X = embeddings_step  # [N, D]
    N, D = X.shape

    p_i = X[i]
    p_j = X[j]
    v = p_j - p_i
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0.0

    v_hat = v / norm  # [D]

    # Center data around p_i and project onto v_hat
    X_centered = X - p_i  # [N, D]
    proj_scores = X_centered @ v_hat  # [N]

    # Total variance across all dimensions
    X_flat = X.reshape(N, D)
    total_var = np.var(X_flat, axis=0).sum()
    if total_var <= 0:
        return 0.0

    var_proj = np.var(proj_scores) * (1.0)  # scalar
    r2 = float(var_proj / total_var)
    return r2


def compute_line_fit_curve(
    batch: ChainBatch,
    metric: str = "cosine",
) -> LineFitCurve:
    """
    For each step t:
      - find the most distant pair of chains in embedding space
      - compute R^2 for a line between them explaining the cloud
    """
    embeddings = batch.embeddings  # [N, T, D]
    N, T, _ = embeddings.shape

    r2_per_step = np.zeros(T, dtype=np.float32)
    extreme_pairs = np.zeros((T, 2), dtype=np.int32)

    for t in range(T):
        emb_t = embeddings[:, t, :]  # [N, D]
        dist_mat = _pairwise_distances_for_step(emb_t, metric=metric)
        i, j = _extreme_pair_indices(dist_mat)
        extreme_pairs[t, 0] = i
        extreme_pairs[t, 1] = j

        r2 = _line_fit_r2_for_step(emb_t, i, j)
        r2_per_step[t] = r2

    return LineFitCurve(
        r2_per_step=r2_per_step,
        extreme_pairs=extreme_pairs,
    )


# -------------------------
# KL divergence between extreme sequences
# -------------------------

def _symmetric_kl(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Symmetric KL divergence between two distributions p and q.
    p, q: [K] arrays that sum to 1.
    Returns scalar.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return float(0.5 * (kl_pq + kl_qp))


def compute_kl_between_extremes(
    batch: ChainBatch,
    line_fit: Optional[LineFitCurve] = None,
) -> KlDivergenceSummary:
    """
    Approximate KL divergence between the two most 'extreme' sequences
    (at the final step), using stored top-k logits.

    Returns:
      - symmetric_kl_curve: [T]
      - symmetric_kl_mean: scalar
    """
    if batch.topk_logits is None or batch.topk_logits.shape[-1] == 0:
        # No logits → cannot compute KL
        return KlDivergenceSummary(
            symmetric_kl_curve=None,
            symmetric_kl_mean=None,
        )

    topk_logits = batch.topk_logits  # [N, T, K]
    N, T, K = topk_logits.shape

    embeddings = batch.embeddings
    # If no LineFitCurve provided, find extremes using final-step distances
    if line_fit is None:
        emb_final = embeddings[:, -1, :]  # [N, D]
        dist_mat = _pairwise_distances_for_step(emb_final, metric="cosine")
        i_extreme, j_extreme = _extreme_pair_indices(dist_mat)
    else:
        # Use extremes from last step stored in line_fit
        i_extreme = int(line_fit.extreme_pairs[-1, 0])
        j_extreme = int(line_fit.extreme_pairs[-1, 1])

    symmetric_kl_curve = np.zeros(T, dtype=np.float32)

    for t in range(T):
        logits_i = topk_logits[i_extreme, t, :]  # [K]
        logits_j = topk_logits[j_extreme, t, :]  # [K]

        # Convert logits → probs (renormalized over top-k)
        p = _softmax_np(logits_i, axis=-1)
        q = _softmax_np(logits_j, axis=-1)

        symmetric_kl_curve[t] = _symmetric_kl(p, q)

    symmetric_kl_mean = float(symmetric_kl_curve.mean())

    return KlDivergenceSummary(
        symmetric_kl_curve=symmetric_kl_curve,
        symmetric_kl_mean=symmetric_kl_mean,
    )


# -------------------------
# Clustering
# -------------------------

def compute_cluster_summary(
    batch: ChainBatch,
    n_clusters: int = 8,
    use_last_step: bool = True,
    random_state: int = 0,
) -> ClusterSummary:
    """
    Simple KMeans clustering over chain representations.

    By default, uses the last-step embedding for each chain: [N, D].
    """
    embeddings = batch.embeddings  # [N, T, D]
    if use_last_step:
        reps = embeddings[:, -1, :]  # [N, D]
    else:
        # mean over time as an alternative
        reps = embeddings.mean(axis=1)  # [N, D]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(reps)  # [N]
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = counts.astype(np.int32)

    meta = {
        "use_last_step": use_last_step,
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_),
    }

    return ClusterSummary(
        num_clusters=n_clusters,
        labels=labels,
        cluster_sizes=cluster_sizes,
        meta=meta,
    )


# -------------------------
# Openness signature
# -------------------------

def build_openness_signature(
    metrics: HorizonMetrics,
) -> OpennessSignature:
    """
    Build a compact feature vector describing the 'openness' of a prompt's horizon.
    """
    mean_entropy = float(metrics.entropy.mean.mean())
    mean_horizon_width = float(metrics.width.mean_dist.mean())
    mean_linearity_r2 = float(metrics.line_fit.r2_per_step.mean())

    mean_symmetric_kl = (
        float(metrics.kl.symmetric_kl_mean)
        if metrics.kl.symmetric_kl_mean is not None
        else None
    )
    num_clusters = (
        metrics.clusters.num_clusters
        if metrics.clusters is not None
        else None
    )

    extra = {
        "entropy_curve": metrics.entropy.mean,
        "width_mean_curve": metrics.width.mean_dist,
        "linearity_r2_curve": metrics.line_fit.r2_per_step,
        "symmetric_kl_curve": metrics.kl.symmetric_kl_curve,
    }

    return OpennessSignature(
        mean_entropy=mean_entropy,
        mean_horizon_width=mean_horizon_width,
        mean_linearity_r2=mean_linearity_r2,
        mean_symmetric_kl=mean_symmetric_kl,
        num_clusters=num_clusters,
        extra=extra,
    )


# -------------------------
# High-level API
# -------------------------

def compute_horizon_metrics(
    batch: ChainBatch,
    cluster_n: int = 8,
) -> HorizonMetrics:
    """
    High-level entrypoint: given a ChainBatch (one prompt),
    compute all openness-related metrics and bundle them in HorizonMetrics.
    """
    # 1) Entropy over time (requires topk_logits)
    entropy = compute_entropy_curve(batch)

    # 2) Horizon width (pairwise distances)
    width = compute_horizon_width_curve(batch, metric="cosine")

    # 3) Line fit (1D vs multi-D structure)
    line_fit = compute_line_fit_curve(batch, metric="cosine")

    # 4) KL between extreme sequences (uses top-k logits)
    kl = compute_kl_between_extremes(batch, line_fit=line_fit)

    # 5) Clustering in embedding space (simple KMeans on last-step embeddings)
    clusters = compute_cluster_summary(batch, n_clusters=cluster_n)

    # 6) Openness signature (summary vector)
    # Build a temporary HorizonMetrics to pass into the signature builder
    tmp_metrics = HorizonMetrics(
        prompt=batch.prompt,
        entropy=entropy,
        width=width,
        line_fit=line_fit,
        kl=kl,
        clusters=clusters,
    )
    signature = build_openness_signature(tmp_metrics)

    # Final HorizonMetrics
    metrics = HorizonMetrics(
        prompt=batch.prompt,
        entropy=entropy,
        width=width,
        line_fit=line_fit,
        kl=kl,
        clusters=clusters,
        signature=signature,
    )

    return metrics
