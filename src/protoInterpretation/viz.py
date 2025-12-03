from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .data_structures import HorizonMetrics, ProjectionResult


# -------------------------
# Time-series plots
# -------------------------

def plot_entropy_curve(
    metrics: HorizonMetrics,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot mean entropy over time (with optional std shading if available).
    """
    if ax is None:
        fig, ax = plt.subplots()

    T = metrics.entropy.mean.shape[0]
    x = np.arange(T)

    ax.plot(x, metrics.entropy.mean, label="Mean entropy")

    if metrics.entropy.std is not None:
        std = metrics.entropy.std
        ax.fill_between(x, metrics.entropy.mean - std, metrics.entropy.mean + std, alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (bits)")
    if title is None:
        title = f"Entropy over time: '{metrics.prompt.text[:40]}'"
    ax.set_title(title)
    ax.legend()

    return ax


def plot_horizon_width(
    metrics: HorizonMetrics,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot mean pairwise distance and max distance over time.
    """
    if ax is None:
        fig, ax = plt.subplots()

    T = metrics.width.mean_dist.shape[0]
    x = np.arange(T)

    ax.plot(x, metrics.width.mean_dist, label="Mean pairwise distance")
    ax.plot(x, metrics.width.max_dist, linestyle="--", label="Max pairwise distance")

    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (cosine)")
    if title is None:
        title = f"Horizon width over time: '{metrics.prompt.text[:40]}'"
    ax.set_title(title)
    ax.legend()

    return ax


# -------------------------
# 2D scatter of horizon
# -------------------------

def plot_step_scatter_2d(
    proj: ProjectionResult,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a 2D scatter of the horizon at a given step using viz_embeddings_2d.
    Optionally color points by integer cluster labels.
    """
    if proj.viz_embeddings_2d is None:
        raise ValueError("viz_embeddings_2d is None; run UMAP or provide 2D embeddings first.")

    X = proj.viz_embeddings_2d  # [N, 2]
    N = X.shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    if labels is not None and labels.shape[0] == N:
        # scatter with labels as colors
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, alpha=0.8)
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Cluster label")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.8)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if title is None:
        title = proj.description or "Horizon projection"
    ax.set_title(title)

    return ax
