from .model import HFModelAdapter as ModelWrapper  # user-friendly alias
from .sampling import sample_chains_for_prompt as sample_chain
from .analysis import compute_horizon_metrics
from .projections import project_step_embeddings, compute_global_umap
from .viz import (
    plot_entropy_curve,
    plot_horizon_width,
    plot_step_scatter_2d,
    plot_metrics_comparison,
    plot_projections_comparison,
    plot_joint_umap,
    create_animation_dataframe,
    plot_animated_umap,
)
from .io import (
    find_runs_directory,
    scan_runs,
    load_run_from_npz,
    load_run_by_name,
    load_embeddings_from_runs,
    save_batch_npz,
    save_metrics_json,
)
from .comparison import compare_runs

# Re-export key dataclasses for convenience
from .data_structures import (
    SamplingConfig,
    PromptSpec,
    ChainBatch,
    HorizonMetrics,
    ProjectionResult,
)
