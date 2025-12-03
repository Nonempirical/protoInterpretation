from .model import HFModelAdapter as ModelWrapper  # user-friendly alias
from .sampling import sample_chains_for_prompt as sample_chain
from .analysis import compute_horizon_metrics
from .projections import project_step_embeddings
from .viz import plot_entropy_curve, plot_horizon_width, plot_step_scatter_2d

# Re-export key dataclasses for convenience
from .data_structures import (
    SamplingConfig,
    PromptSpec,
    ChainBatch,
    HorizonMetrics,
    ProjectionResult,
)
