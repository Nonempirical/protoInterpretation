from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness

try:
    import umap  # type: ignore
except ImportError:
    umap = None  # We'll handle this gracefully

from .data_structures import ChainBatch, ProjectionResult


@dataclass
class PCAResult:
    pca: PCA
    embeddings: np.ndarray  # [N, D_pca]


def fit_pca(
    X: np.ndarray,
    n_components: int = 50,
) -> PCAResult:
    """
    Fit PCA on X and return reduced embeddings.
    X: [N, D]
    """
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=0)
    embeddings = pca.fit_transform(X)
    return PCAResult(pca=pca, embeddings=embeddings)


def apply_pca(
    pca: PCA,
    X: np.ndarray,
) -> np.ndarray:
    """
    Apply an existing PCA model to new data.
    """
    return pca.transform(X)


def _compute_dr_diagnostics(
    base_embeddings: np.ndarray,
    viz_embeddings: np.ndarray,
    n_neighbors: int = 15,
) -> Dict[str, Any]:
    """
    Compute a few simple diagnostic metrics for dimensionality reduction.
    """
    diagnostics: Dict[str, Any] = {}

    # Trustworthiness (sklearn implementation)
    try:
        tw = trustworthiness(
            base_embeddings,
            viz_embeddings,
            n_neighbors=n_neighbors,
            metric="euclidean",
        )
        diagnostics["trustworthiness"] = float(tw)
    except Exception as e:
        diagnostics["trustworthiness_error"] = str(e)

    # Distance correlation (Pearson correlation between flattened distance matrices)
    try:
        dist_high = pairwise_distances(base_embeddings, metric="euclidean")
        dist_low = pairwise_distances(viz_embeddings, metric="euclidean")

        # Use upper triangle to avoid double-counting / diagonals
        triu = np.triu_indices(dist_high.shape[0], k=1)
        dh = dist_high[triu].ravel()
        dl = dist_low[triu].ravel()

        if dh.size > 0:
            dh_mean = dh.mean()
            dl_mean = dl.mean()
            num = np.sum((dh - dh_mean) * (dl - dl_mean))
            den = np.sqrt(np.sum((dh - dh_mean) ** 2) * np.sum((dl - dl_mean) ** 2))
            corr = float(num / den) if den > 0 else 0.0
        else:
            corr = 0.0

        diagnostics["distance_correlation"] = corr
    except Exception as e:
        diagnostics["distance_correlation_error"] = str(e)

    return diagnostics


def project_step_embeddings(
    batch: ChainBatch,
    step: int = -1,
    pca_components: int = 50,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 0,
) -> ProjectionResult:
    """
    Project the embeddings at a single step t into PCA space (for analysis)
    and UMAP 2D space (for visualization).

    Returns a ProjectionResult containing:
      - base_embeddings: [N, D] original embeddings at step t
      - pca_embeddings:  [N, D_pca]
      - viz_embeddings_2d: [N, 2] UMAP projection
      - diagnostics: DR quality metrics
    """
    embeddings = batch.embeddings  # [N, T, D]
    N, T, D = embeddings.shape

    if step < 0:
        step = T + step
    if not (0 <= step < T):
        raise ValueError(f"step {step} out of range [0, {T})")

    base = embeddings[:, step, :]  # [N, D]

    # PCA to mid-dimensional space
    pca_res = fit_pca(base, n_components=pca_components)
    pca_emb = pca_res.embeddings  # [N, D_pca]

    if umap is None:
        # UMAP not available; just return PCA and diagnostics based on PCA vs original
        diagnostics = _compute_dr_diagnostics(base, pca_emb)
        return ProjectionResult(
            base_embeddings=base,
            pca_embeddings=pca_emb,
            viz_embeddings_2d=None,
            viz_embeddings_3d=None,
            diagnostics=diagnostics,
            description=f"Step {step} PCA-only projection",
            meta={"step": step},
        )

    # UMAP 2D projection from PCA space
    umap_model = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=umap_random_state,
    )
    viz_2d = umap_model.fit_transform(pca_emb)  # [N, 2]

    diagnostics = _compute_dr_diagnostics(base, viz_2d)

    return ProjectionResult(
        base_embeddings=base,
        pca_embeddings=pca_emb,
        viz_embeddings_2d=viz_2d,
        viz_embeddings_3d=None,
        diagnostics=diagnostics,
        description=f"Step {step} PCA+UMAP projection",
        meta={
            "step": step,
            "pca_components": pca_components,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
        },
    )
