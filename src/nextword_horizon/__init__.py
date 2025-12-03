from .model import HFModelAdapter as ModelWrapper  # user-friendly alias
from .sampling import sample_chains_for_prompt as sample_chain
from .analysis import compute_horizon_metrics

# Re-export key dataclasses for convenience
from .data_structures import (
    SamplingConfig,
    PromptSpec,
    ChainBatch,
    HorizonMetrics,
    ProjectionResult,
)
