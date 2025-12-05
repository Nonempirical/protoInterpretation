from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .data_structures import ChainBatch, HorizonMetrics
from .projections import project_step_embeddings
from .viz import (
    plot_metrics_comparison,
    plot_projections_comparison,
    plot_joint_umap,
)


def compare_runs(
    batch_A: ChainBatch,
    batch_B: ChainBatch,
    metrics_A: HorizonMetrics,
    metrics_B: HorizonMetrics,
    step: int = -1,
    label_A: str = "A",
    label_B: str = "B",
    show_metrics: bool = True,
    show_projections: bool = True,
    show_joint_umap: bool = True,
    synchronize_projection_axes: bool = True,
) -> dict:
    """
    Comprehensive comparison of two runs with visualizations.
    
    Args:
        batch_A: First ChainBatch
        batch_B: Second ChainBatch
        metrics_A: HorizonMetrics for first batch
        metrics_B: HorizonMetrics for second batch
        step: Step to use for projections (default: -1 for last step)
        label_A: Label for first run
        label_B: Label for second run
        show_metrics: If True, show entropy and width comparison
        show_projections: If True, show side-by-side UMAP projections
        show_joint_umap: If True, show joint UMAP of both runs
        synchronize_projection_axes: If True, synchronize axes in projection plots
    
    Returns:
        Dictionary containing:
            - 'step': The step used for projections
            - 'proj_A': ProjectionResult for batch A
            - 'proj_B': ProjectionResult for batch B
            - 'figures': List of matplotlib figures created
    """
    # Determine valid step range
    T_A = metrics_A.entropy.mean.shape[0]
    T_B = metrics_B.entropy.mean.shape[0]
    max_T = min(T_A, T_B)
    
    if step < 0:
        step = max_T + step
    step = min(step, max_T - 1)
    
    results = {
        'step': step,
        'proj_A': None,
        'proj_B': None,
        'figures': [],
    }
    
    # Metrics comparison
    if show_metrics:
        fig_metrics, _ = plot_metrics_comparison(
            metrics_A, metrics_B,
            label_A=label_A, label_B=label_B
        )
        results['figures'].append(fig_metrics)
        plt.show()
    
    # Projections comparison
    if show_projections:
        proj_A = project_step_embeddings(batch_A, step=step)
        proj_B = project_step_embeddings(batch_B, step=step)
        results['proj_A'] = proj_A
        results['proj_B'] = proj_B
        
        fig_proj, _ = plot_projections_comparison(
            proj_A, proj_B,
            label_A=label_A, label_B=label_B,
            title_A=f"{label_A} @ step {step}",
            title_B=f"{label_B} @ step {step}",
            synchronize_axes=synchronize_projection_axes,
        )
        results['figures'].append(fig_proj)
        plt.show()
    
    # Joint UMAP
    if show_joint_umap:
        E_A = batch_A.embeddings[:, step, :]  # [N_A, D]
        E_B = batch_B.embeddings[:, step, :]  # [N_B, D]
        
        fig_joint, _, _ = plot_joint_umap(
            E_A, E_B,
            label_A=label_A, label_B=label_B,
            title=f"Joint UMAP @ step {step}",
        )
        results['figures'].append(fig_joint)
        plt.show()
    
    return results

