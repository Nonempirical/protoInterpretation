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


# -------------------------
# Comparison plots
# -------------------------

def plot_metrics_comparison(
    metrics_A: HorizonMetrics,
    metrics_B: HorizonMetrics,
    label_A: str = "A",
    label_B: str = "B",
    figsize: tuple[int, int] = (12, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot side-by-side comparison of entropy and horizon width curves.
    
    Args:
        metrics_A: First HorizonMetrics object
        metrics_B: Second HorizonMetrics object
        label_A: Label for first metrics
        label_B: Label for second metrics
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (figure, axes array)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    T_A = metrics_A.entropy.mean.shape[0]
    T_B = metrics_B.entropy.mean.shape[0]
    x_A = np.arange(T_A)
    x_B = np.arange(T_B)
    
    # Entropy comparison
    axes[0].plot(x_A, metrics_A.entropy.mean, label=f"{label_A}", linewidth=2)
    axes[0].plot(x_B, metrics_B.entropy.mean, label=f"{label_B}", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_title("Entropy over time")
    axes[0].legend()
    
    # Horizon width comparison
    axes[1].plot(x_A, metrics_A.width.mean_dist, label=f"{label_A}", linewidth=2)
    axes[1].plot(x_B, metrics_B.width.mean_dist, label=f"{label_B}", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean pairwise distance (cosine)")
    axes[1].set_title("Horizon width over time")
    axes[1].legend()
    
    plt.tight_layout()
    return fig, axes


def plot_projections_comparison(
    proj_A: ProjectionResult,
    proj_B: ProjectionResult,
    label_A: str = "A",
    label_B: str = "B",
    title_A: Optional[str] = None,
    title_B: Optional[str] = None,
    figsize: tuple[int, int] = (12, 5),
    synchronize_axes: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot side-by-side comparison of two projections with synchronized axes.
    
    Args:
        proj_A: First ProjectionResult
        proj_B: Second ProjectionResult
        label_A: Label for first projection
        label_B: Label for second projection
        title_A: Optional title for first subplot
        title_B: Optional title for second subplot
        figsize: Figure size (width, height)
        synchronize_axes: If True, synchronize axis ranges across both plots
    
    Returns:
        Tuple of (figure, axes array)
    """
    if proj_A.viz_embeddings_2d is None or proj_B.viz_embeddings_2d is None:
        raise ValueError("Both projections must have viz_embeddings_2d")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    X_A = proj_A.viz_embeddings_2d
    X_B = proj_B.viz_embeddings_2d
    
    # Plot first projection
    plot_step_scatter_2d(
        proj_A,
        labels=None,
        ax=axes[0],
        title=title_A or f"{label_A}",
    )
    
    # Plot second projection
    plot_step_scatter_2d(
        proj_B,
        labels=None,
        ax=axes[1],
        title=title_B or f"{label_B}",
    )
    
    # Synchronize axis ranges if requested
    if synchronize_axes:
        x_min = min(X_A[:, 0].min(), X_B[:, 0].min())
        x_max = max(X_A[:, 0].max(), X_B[:, 0].max())
        y_min = min(X_A[:, 1].min(), X_B[:, 1].min())
        y_max = max(X_A[:, 1].max(), X_B[:, 1].max())
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig, axes


def plot_joint_umap(
    embeddings_A: np.ndarray,
    embeddings_B: np.ndarray,
    label_A: str = "A",
    label_B: str = "B",
    pca_components: int = 50,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 0,
    figsize: tuple[int, int] = (6, 5),
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Compute and plot a joint UMAP projection of two sets of embeddings.
    
    Args:
        embeddings_A: Embeddings for first set [N_A, D]
        embeddings_B: Embeddings for second set [N_B, D]
        label_A: Label for first set
        label_B: Label for second set
        pca_components: Number of PCA components
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
        umap_random_state: Random state for reproducibility
        figsize: Figure size (width, height)
        title: Optional plot title
    
    Returns:
        Tuple of (figure, axes, joint_2d_embeddings)
    """
    try:
        from sklearn.decomposition import PCA
        import umap
    except ImportError:
        raise ImportError("sklearn and umap-learn are required for joint UMAP visualization")
    
    # Stack embeddings
    E = np.vstack([embeddings_A, embeddings_B])  # [N_A + N_B, D]
    N_A = embeddings_A.shape[0]
    
    # PCA → UMAP
    n_pca = min(pca_components, E.shape[1], E.shape[0])
    pca = PCA(n_components=n_pca, random_state=0)
    E_pca = pca.fit_transform(E)
    
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=umap_random_state,
    )
    E_2d = reducer.fit_transform(E_pca)  # [N_A + N_B, 2]
    
    E_A_2d = E_2d[:N_A]
    E_B_2d = E_2d[N_A:]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(E_A_2d[:, 0], E_A_2d[:, 1], s=15, alpha=0.8, label=label_A)
    ax.scatter(E_B_2d[:, 0], E_B_2d[:, 1], s=15, alpha=0.8, label=label_B)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if title is None:
        title = "Joint UMAP"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    return fig, ax, E_2d


# -------------------------
# Plotly animations
# -------------------------

try:
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    pd = None


def create_animation_dataframe(
    embeddings_2d: np.ndarray,
    run_labels: list[str],
    step_labels: list[int],
    margin_factor: float = 0.05,
) -> "pd.DataFrame":
    """
    Create a pandas DataFrame for Plotly animation.
    
    Args:
        embeddings_2d: 2D embeddings [total_points, 2]
        run_labels: List of run names for each point
        step_labels: List of step indices for each point
        margin_factor: Factor for adding margins to axis ranges
    
    Returns:
        pandas DataFrame with columns: x, y, run, step
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly and pandas are required for animation DataFrame creation")
    
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "run": run_labels,
        "step": step_labels,
    })
    
    # Add margins to axis ranges
    x_margin = (df["x"].max() - df["x"].min()) * margin_factor
    y_margin = (df["y"].max() - df["y"].min()) * margin_factor
    
    df.attrs["x_range"] = [df["x"].min() - x_margin, df["x"].max() + x_margin]
    df.attrs["y_range"] = [df["y"].min() - y_margin, df["y"].max() + y_margin]
    
    return df


def plot_animated_umap(
    embeddings_2d: np.ndarray,
    run_labels: list[str],
    step_labels: list[int],
    title: str = "Global UMAP horizon movie",
    width: int = 700,
    height: int = 600,
    marker_size: int = 5,
    opacity: float = 0.8,
    margin_factor: float = 0.05,
) -> "px.scatter":
    """
    Create an animated Plotly scatter plot showing how embeddings evolve over steps.
    
    Args:
        embeddings_2d: 2D embeddings [total_points, 2]
        run_labels: List of run names for each point
        step_labels: List of step indices for each point
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        marker_size: Size of scatter points
        opacity: Opacity of scatter points
        margin_factor: Factor for adding margins to axis ranges
    
    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly and pandas are required for animated UMAP visualization")
    
    df = create_animation_dataframe(embeddings_2d, run_labels, step_labels, margin_factor=margin_factor)
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="run",
        animation_frame="step",
        range_x=df.attrs["x_range"],
        range_y=df.attrs["y_range"],
        title=title,
        opacity=opacity,
    )
    
    # Customize appearance
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        width=width,
        height=height,
        legend=dict(font=dict(size=10)),
    )
    
    return fig