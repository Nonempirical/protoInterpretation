from .model import HFModelAdapter as ModelWrapper  # user-friendly alias
from .sampling import sample_chains_for_prompt as sample_chain
from .analysis import compute_horizon_metrics, compute_cumulative_embeddings
from .projections import project_step_embeddings, compute_global_umap
from .viz import (
    plot_entropy_curve,
    plot_horizon_width,
    plot_entropy_comparison,
    plot_horizon_width_comparison,
    plot_final_step_umap_across_runs,
    plot_global_umap_facets_by_step,
    save_figures,
    save_open_figures,
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
    slugify_prompt,
    run_name_to_display,
    run_name_to_filename,
    save_horizon_run,
    save_horizon_run_from_prompt,
    run_prompts_and_save,
    summarize_signature,
)
from .comparison import compare_runs
from .research_guide import print_research_guide

# Re-export key dataclasses for convenience
from .data_structures import (
    SamplingConfig,
    PromptSpec,
    ChainBatch,
    HorizonMetrics,
    ProjectionResult,
)
