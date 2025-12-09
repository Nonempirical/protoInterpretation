from __future__ import annotations

import json
import os
from typing import Dict, Optional, List, Tuple

import numpy as np

from .data_structures import ChainBatch, PromptSpec, HorizonMetrics
from .analysis import compute_horizon_metrics


def find_runs_directory(
    base_path: str,
    folder_name: str = "protoInterpretation-runs",
    max_depth: int = 10,
) -> Optional[str]:
    """
    Search for a runs directory starting from base_path.
    
    Args:
        base_path: Root directory to search from (e.g., "/content/drive/MyDrive")
        folder_name: Name of the folder to find
        max_depth: Maximum depth to search (to avoid infinite loops)
    
    Returns:
        Path to the runs directory, or None if not found
    """
    if not os.path.exists(base_path):
        return None
    
    for current_root, dirs, files in os.walk(base_path):
        # Limit depth
        depth = current_root[len(base_path):].count(os.sep)
        if depth > max_depth:
            continue
        
        if folder_name in dirs:
            return os.path.join(current_root, folder_name)
    
    return None


def scan_runs(runs_dir: str) -> Dict[str, str]:
    """
    Scan a runs directory for subdirectories containing batch.npz files.
    
    Args:
        runs_dir: Path to the runs directory
    
    Returns:
        Dictionary mapping run names to their directory paths
    """
    runs = {}
    
    if not os.path.exists(runs_dir):
        return runs
    
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "batch.npz")):
            runs[name] = run_path
    
    return runs


def load_run_from_npz(run_path: str, compute_metrics: bool = True) -> Tuple[ChainBatch, Optional[HorizonMetrics]]:
    """
    Load a ChainBatch from a batch.npz file.
    
    Args:
        run_path: Path to the run directory containing batch.npz
        compute_metrics: If True, also compute and return HorizonMetrics
    
    Returns:
        Tuple of (ChainBatch, HorizonMetrics or None)
    """
    batch_path = os.path.join(run_path, "batch.npz")
    
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"batch.npz not found in {run_path}")
    
    data = np.load(batch_path, allow_pickle=True)
    
    # Extract prompt information
    prompt_text = str(data["prompt_text"])
    raw_label = data["prompt_label"]
    
    if isinstance(raw_label, np.ndarray):
        raw_label = raw_label.item()
    
    prompt_label = str(raw_label) if raw_label != "" else None
    
    # Handle optional arrays (empty arrays become None)
    topk_token_ids = data["topk_token_ids"]
    if topk_token_ids.size == 0:
        topk_token_ids = None
    
    topk_logits = data["topk_logits"]
    if topk_logits.size == 0:
        topk_logits = None
    
    # Handle text sequences (may not exist in older files)
    text_sequences = None
    if "text_sequences" in data:
        text_seq_array = data["text_sequences"]
        if text_seq_array.size > 0:
            # Convert numpy array of objects to list of strings
            text_sequences = [str(seq) for seq in text_seq_array]
    
    # Create ChainBatch
    batch = ChainBatch(
        prompt=PromptSpec(text=prompt_text, label=prompt_label),
        prompt_token_ids=data["prompt_token_ids"],
        token_ids=data["token_ids"],
        embeddings=data["embeddings"],
        topk_token_ids=topk_token_ids,
        topk_logits=topk_logits,
        step_mask=data["step_mask"],
        text_sequences=text_sequences,
        meta={"run_dir": run_path},
    )
    
    # Optionally compute metrics
    metrics = None
    if compute_metrics:
        metrics = compute_horizon_metrics(batch)
    
    return batch, metrics


def load_run_by_name(run_name: str, runs: Dict[str, str], compute_metrics: bool = True) -> Tuple[ChainBatch, Optional[HorizonMetrics]]:
    """
    Load a run by name from a runs dictionary.
    
    Args:
        run_name: Name of the run
        runs: Dictionary mapping run names to paths (from scan_runs)
        compute_metrics: If True, also compute and return HorizonMetrics
    
    Returns:
        Tuple of (ChainBatch, HorizonMetrics or None)
    """
    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in runs dictionary")
    
    return load_run_from_npz(runs[run_name], compute_metrics=compute_metrics)


def load_embeddings_from_runs(
    run_names: List[str],
    runs: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Load embeddings from multiple runs for animation/comparison.
    
    Args:
        run_names: List of run names to load
        runs: Dictionary mapping run names to paths (from scan_runs)
    
    Returns:
        Tuple of:
            - Dictionary mapping run names to embeddings [N, T, D]
            - Minimum T across all runs
            - Embedding dimension D
    """
    emb_per_run = {}
    min_T = None
    D = None
    
    for name in run_names:
        if name not in runs:
            raise ValueError(f"Run '{name}' not found in runs dictionary")
        
        batch_path = os.path.join(runs[name], "batch.npz")
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"batch.npz not found for run '{name}'")
        
        data = np.load(batch_path, allow_pickle=True)
        emb = data["embeddings"]  # [N, T, D]
        emb_per_run[name] = emb
        
        N, T, D_ = emb.shape
        D = D_ if D is None else D
        min_T = T if min_T is None else min(min_T, T)
    
    return emb_per_run, min_T, D


# -------------------------
# Save functions
# -------------------------

def save_batch_npz(
    batch: ChainBatch,
    output_path: str,
) -> None:
    """
    Save ChainBatch to batch.npz file (numeric data only, no metadata).
    
    Args:
        batch: ChainBatch to save
        output_path: Path where batch.npz will be saved
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save only numeric arrays (no meta)
    # Text sequences are saved as a structured array
    text_sequences_array = np.array(
        batch.text_sequences if batch.text_sequences else [],
        dtype=object
    )
    
    np.savez(
        output_path,
        prompt_text=batch.prompt.text,
        prompt_label=batch.prompt.label if batch.prompt.label else "",
        prompt_token_ids=batch.prompt_token_ids,
        token_ids=batch.token_ids,
        embeddings=batch.embeddings,
        topk_token_ids=batch.topk_token_ids if batch.topk_token_ids is not None else np.array([]),
        topk_logits=batch.topk_logits if batch.topk_logits is not None else np.array([]),
        step_mask=batch.step_mask if batch.step_mask is not None else np.array([]),
        text_sequences=text_sequences_array,
    )


def save_metrics_json(
    metrics: HorizonMetrics,
    output_path: str,
    batch_meta: Optional[Dict] = None,
) -> None:
    """
    Save HorizonMetrics to metrics.json file, including meta from batch.
    
    Args:
        metrics: HorizonMetrics to save
        output_path: Path where metrics.json will be saved
        batch_meta: Optional dict containing batch.meta to include in saved metrics
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Convert metrics to dict
    metrics_data = {
        "prompt": {
            "text": metrics.prompt.text,
            "label": metrics.prompt.label,
        },
        "entropy": {
            "mean": metrics.entropy.mean.tolist(),
            "std": metrics.entropy.std.tolist() if metrics.entropy.std is not None else None,
        },
        "width": {
            "max_dist": metrics.width.max_dist.tolist(),
            "mean_dist": metrics.width.mean_dist.tolist(),
            "p95_dist": metrics.width.p95_dist.tolist() if metrics.width.p95_dist is not None else None,
        },
        "line_fit": {
            "r2_per_step": metrics.line_fit.r2_per_step.tolist(),
        },
        "kl": {
            "symmetric_kl_curve": metrics.kl.symmetric_kl_curve.tolist() if metrics.kl.symmetric_kl_curve is not None else None,
            "symmetric_kl_mean": metrics.kl.symmetric_kl_mean,
        },
    }
    
    # Add clusters if available
    if metrics.clusters is not None:
        metrics_data["clusters"] = {
            "num_clusters": metrics.clusters.num_clusters,
            "labels": metrics.clusters.labels.tolist(),
            "cluster_sizes": metrics.clusters.cluster_sizes.tolist() if metrics.clusters.cluster_sizes is not None else None,
            "meta": metrics.clusters.meta,
        }
    
    # Add signature if available
    if metrics.signature is not None:
        metrics_data["signature"] = {
            "mean_entropy": metrics.signature.mean_entropy,
            "mean_horizon_width": metrics.signature.mean_horizon_width,
            "mean_linearity_r2": metrics.signature.mean_linearity_r2,
            "mean_symmetric_kl": metrics.signature.mean_symmetric_kl,
            "num_clusters": metrics.signature.num_clusters,
        }
    
    # Add meta from batch if provided
    if batch_meta:
        metrics_data["meta"] = batch_meta
    elif metrics.meta:
        metrics_data["meta"] = metrics.meta
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

