from __future__ import annotations

import os
import re
from typing import Optional, Dict, Sequence, Iterable, Tuple, Mapping, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.gridspec import GridSpec

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
    Plot horizon width components over time.

    Shows (when available):
    - mean_dist: mean pairwise cosine distance
    - p95_dist: 95th percentile pairwise cosine distance
    - max_dist: max pairwise cosine distance
    """
    if ax is None:
        fig, ax = plt.subplots()

    T = metrics.width.mean_dist.shape[0]
    x = np.arange(T)

    ax.plot(x, metrics.width.mean_dist, label="Mean", linestyle="-")
    if metrics.width.p95_dist is not None:
        ax.plot(x, metrics.width.p95_dist, label="P95", linestyle=":")
    ax.plot(x, metrics.width.max_dist, label="Max", linestyle="--")

    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (cosine)")
    if title is None:
        title = f"Horizon width over time: '{metrics.prompt.text[:40]}'"
    ax.set_title(title)
    ax.legend()

    return ax


def plot_entropy_comparison(
    metrics_per_run: Mapping[str, HorizonMetrics],
    labels: Optional[Mapping[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Entropy over time",
) -> plt.Axes:
    """
    Overlay entropy curves for multiple runs in a single axis.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.cm.tab10.colors
    for i, (run_name, metrics) in enumerate(metrics_per_run.items()):
        x = np.arange(metrics.entropy.mean.shape[0])
        label = labels.get(run_name, run_name) if labels is not None else run_name
        ax.plot(x, metrics.entropy.mean, label=label, linewidth=2, color=colors[i % len(colors)])

    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_horizon_width_comparison(
    metrics_per_run: Mapping[str, HorizonMetrics],
    labels: Optional[Mapping[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Horizon width over time",
    show: Sequence[str] = ("mean", "p95", "max"),
) -> plt.Axes:
    """
    Overlay horizon width curves for multiple runs, with 3 components per run.

    The intent is to keep one color per run and use line styles for components:
    - mean: solid
    - p95: dotted
    - max: dashed
    """
    if ax is None:
        _, ax = plt.subplots()

    styles = {"mean": "-", "p95": ":", "max": "--"}
    colors = plt.cm.tab10.colors

    for i, (run_name, metrics) in enumerate(metrics_per_run.items()):
        c = colors[i % len(colors)]
        base_label = labels.get(run_name, run_name) if labels is not None else run_name
        x = np.arange(metrics.width.mean_dist.shape[0])

        if "mean" in show:
            ax.plot(x, metrics.width.mean_dist, color=c, linestyle=styles["mean"], linewidth=2, label=f"{base_label} (mean)")
        if "p95" in show and metrics.width.p95_dist is not None:
            ax.plot(x, metrics.width.p95_dist, color=c, linestyle=styles["p95"], linewidth=2, label=f"{base_label} (p95)")
        if "max" in show:
            ax.plot(x, metrics.width.max_dist, color=c, linestyle=styles["max"], linewidth=2, label=f"{base_label} (max)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (cosine)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_final_step_umap_across_runs(
    embeddings_per_run: Mapping[str, np.ndarray],
    *,
    step: int = -1,
    labels: Optional[Mapping[str, str]] = None,
    pca_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 42,
    l2_normalize: bool = False,
    umap_metric: str = "cosine",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    point_size: int = 25,
    alpha: float = 0.6,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray, Sequence[str], np.ndarray]:
    """
    Compute a joint UMAP for a single step (default: last step) across multiple runs,
    and plot all chains on one shared map (colored by run).

    This mirrors a common Colab visualization pattern:
      - One UMAP space for all runs at the final timestep
      - Color encodes run/prompt

    Returns:
        (fig, ax, embeddings_2d, run_labels, chain_indices)
    """
    # Local import to avoid circular imports at module import-time
    from .projections import compute_global_umap

    # Slice each run to a single-step [N, 1, D] tensor
    sliced: Dict[str, np.ndarray] = {}
    for name, emb in embeddings_per_run.items():
        if emb.ndim != 3:
            raise ValueError(f"Expected embeddings [N,T,D] for run '{name}', got shape {emb.shape}")
        T = emb.shape[1]
        step_idx = T + step if step < 0 else step
        if not (0 <= step_idx < T):
            raise ValueError(f"step {step} out of range for run '{name}' with T={T}")
        sliced[name] = emb[:, step_idx : step_idx + 1, :]

    E_2d, run_labels, _step_labels, chain_indices = compute_global_umap(
        sliced,
        max_steps=1,
        pca_components=pca_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
        l2_normalize=l2_normalize,
        umap_metric=umap_metric,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    colors = plt.cm.tab10.colors
    run_labels_arr = np.asarray(run_labels)
    runs = list(dict.fromkeys(run_labels))

    for i, run_name in enumerate(runs):
        mask = run_labels_arr == run_name
        disp = labels.get(run_name, run_name) if labels is not None else run_name
        ax.scatter(
            E_2d[mask, 0],
            E_2d[mask, 1],
            label=disp,
            color=colors[i % len(colors)],
            s=point_size,
            alpha=alpha,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP @ step {step}")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax, E_2d, run_labels, chain_indices


def plot_global_umap_facets_by_step(
    embeddings_2d: np.ndarray,
    run_labels: Sequence[str],
    step_labels: Sequence[int],
    *,
    run_order: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    facet_col_wrap: int = 2,
    cmap_name: str = "plasma",
    point_size: int = 8,
    alpha: float = 0.85,
    title: str = "Global UMAP colored by timestep",
) -> plt.Figure:
    """
    Matplotlib equivalent of the common Plotly facet UMAP:
      - One panel per run
      - Point color encodes step index
      - Shared axis limits across panels + shared colorbar
    """
    run_labels_arr = np.asarray(run_labels)
    step_labels_arr = np.asarray(step_labels)

    runs = list(run_order) if run_order is not None else list(dict.fromkeys(run_labels))
    n_runs = len(runs)
    n_cols = max(1, int(facet_col_wrap))
    n_rows = int(np.ceil(n_runs / n_cols))

    # Shared axis limits (with margins)
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    x_min, x_max = x.min() - x_margin, x.max() + x_margin
    y_min, y_max = y.min() - y_margin, y.max() + y_margin

    # Color mapping for step
    vmin = int(step_labels_arr.min()) if step_labels_arr.size else 0
    vmax = int(step_labels_arr.max()) if step_labels_arr.size else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = getattr(cm, cmap_name, cm.plasma)

    fig = plt.figure(figsize=(6 * n_cols + 1, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.3, hspace=0.3)

    last_scatter = None
    for idx, run_name in enumerate(runs):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        mask = run_labels_arr == run_name
        last_scatter = ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=step_labels_arr[mask],
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        disp = labels.get(run_name, run_name) if labels is not None else run_name
        ax.set_title(disp, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.grid(True, alpha=0.2)

    # Turn off empty panels
    for idx in range(n_runs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.add_subplot(gs[row, col]).axis("off")

    # Colorbar
    if last_scatter is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(last_scatter, cax=cbar_ax)
        cbar.set_label("Timestep", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# -------------------------
# Saving figures
# -------------------------

def _slugify_filename_stem(text: str, max_len: int = 80) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s[:max_len].rstrip("_")
    return s or "figure"


def save_figures(
    figures: Mapping[str, plt.Figure],
    output_dir: str,
    *,
    dpi: int = 150,
    facecolor: str = "white",
    bbox_inches: str = "tight",
    close: bool = False,
) -> List[str]:
    """
    Save multiple matplotlib figures as individual PNG files.

    Args:
        figures: Mapping {name -> fig}. The name is used as the filename stem.
        output_dir: Directory where PNGs are written.
        dpi: PNG DPI.
        facecolor: Figure facecolor to use when saving.
        bbox_inches: Passed to fig.savefig (default: "tight").
        close: If True, close each figure after saving.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    for name, fig in figures.items():
        stem = _slugify_filename_stem(name)
        path = os.path.join(output_dir, f"{stem}.png")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
        saved_paths.append(path)
        if close:
            plt.close(fig)

    return saved_paths


def save_open_figures(
    output_dir: str,
    *,
    prefix: str = "figure",
    dpi: int = 150,
    facecolor: str = "white",
    bbox_inches: str = "tight",
    close: bool = False,
) -> List[str]:
    """
    Save all currently open matplotlib figures (useful if you didn't keep fig variables).

    Figures are saved as: {prefix}_{fig_number}.png
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(output_dir, f"{_slugify_filename_stem(prefix)}_{num}.png")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
        saved_paths.append(path)
        if close:
            plt.close(fig)

    return saved_paths


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
    umap_random_state: int = 42,
    l2_normalize: bool = False,
    umap_metric: str = "cosine",
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
    
    def _l2_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + eps)

    # Stack embeddings
    E = np.vstack([embeddings_A, embeddings_B])  # [N_A + N_B, D]
    N_A = embeddings_A.shape[0]
    if l2_normalize:
        E = _l2_norm_rows(E)
    
    # PCA → UMAP
    n_pca = min(pca_components, E.shape[1], E.shape[0])
    pca = PCA(n_components=n_pca, random_state=0)
    E_pca = pca.fit_transform(E)
    if l2_normalize:
        # PCA breaks unit norms; renormalize for cosine-aligned Euclidean UMAP
        E_pca = _l2_norm_rows(E_pca)
    
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=umap_random_state,
        metric=umap_metric,
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
    animation_frame_duration: int = 200,
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
        animation_frame_duration: Duration of each frame in milliseconds (lower = faster)
    
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
    
    # Set animation speed - CRITICAL: This actually controls playback speed
    if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
        btn = fig.layout.updatemenus[0].buttons[0]
        btn.args[1]["frame"]["duration"] = animation_frame_duration
        btn.args[1]["transition"]["duration"] = 0
    
    if fig.layout.sliders and len(fig.layout.sliders) > 0:
        fig.layout.sliders[0]["transition"]["duration"] = 0
    
    return fig