from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .model import HFModelAdapter
from .data_structures import SamplingConfig, PromptSpec, ChainBatch


# -------------------------
# Sampling utilities
# -------------------------

def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return logits
    return logits / temperature


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p).

    This is a simplified variant of the HuggingFace implementation.
    logits: [B, V]
    """
    top_k = int(top_k)
    top_p = float(top_p)

    # clone to avoid modifying in-place
    scores = logits.clone()

    # Top-k
    if top_k > 0 and top_k < scores.size(-1):
        values, _ = torch.topk(scores, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        scores = torch.where(scores < min_values, torch.full_like(scores, float("-inf")), scores)

    # Top-p (nucleus)
    if top_p < 1.0:
        # sort by descending logit
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(sorted_scores, dim=-1)

        cumulative_probs = probs.cumsum(dim=-1)
        # tokens to remove: cumulative prob > top_p
        cutoff = cumulative_probs > top_p

        # shift so we always keep at least one token
        cutoff[:, 0] = False

        # set logits of removed indices to -inf
        sorted_scores = torch.where(cutoff, torch.full_like(sorted_scores, float("-inf")), sorted_scores)

        # map back to original order
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

    return scores


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Sample next tokens from logits with temperature + top-k/top-p.

    logits: [B, V]
    returns: next_token_ids: [B]
    """
    scores = _apply_temperature(logits, temperature)
    scores = _top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)
    # multinomial sampling per row
    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_token_ids


# -------------------------
# Main API
# -------------------------

def sample_chains_for_prompt(
    model: HFModelAdapter,
    prompt: str | PromptSpec,
    sampling_cfg: SamplingConfig,
) -> ChainBatch:
    """
    Generate multiple independent chains from a single prompt.

    Returns a ChainBatch containing:
      - token_ids:   [N, T]
      - embeddings:  [N, T, D]  (last-token hidden per step)
      - topk_logits: [N, T, K]
      - topk_token_ids: [N, T, K]
    """
    _set_seed(sampling_cfg.seed)

    # Normalize prompt into PromptSpec
    if isinstance(prompt, str):
        prompt_spec = PromptSpec(text=prompt)
    else:
        prompt_spec = prompt

    # Encode prompt once
    prompt_token_ids_list: List[int] = model.encode(prompt_spec.text)
    prompt_token_ids_np = np.array(prompt_token_ids_list, dtype=np.int64)

    num_chains = sampling_cfg.num_chains
    max_steps = sampling_cfg.max_steps

    # Initialize batch of token sequences: each chain starts with the same prompt
    batch_token_ids: List[List[int]] = [
        list(prompt_token_ids_list) for _ in range(num_chains)
    ]

    # We'll collect per-step tensors then stack at the end
    embeddings_steps: List[torch.Tensor] = []  # each: [N, D]
    topk_ids_steps: List[torch.Tensor] = []    # each: [N, K]
    topk_logits_steps: List[torch.Tensor] = [] # each: [N, K]
    token_ids_steps: List[torch.Tensor] = []   # each: [N]

    # Main generation loop
    for step in range(max_steps):
        # 1) Get logits + embedding for current sequences
        logits_next, last_hidden = model.get_logits_and_embeddings(batch_token_ids)
        # logits_next: [N, V]
        # last_hidden: [N, D]

        # 2) Store embeddings
        embeddings_steps.append(last_hidden.detach().cpu())

        # 3) Store top-k logits if requested
        if sampling_cfg.store_topk_logits > 0:
            k = min(sampling_cfg.store_topk_logits, logits_next.shape[-1])
            topk_logits, topk_ids = torch.topk(logits_next, k=k, dim=-1)
            topk_ids_steps.append(topk_ids.detach().cpu())
            topk_logits_steps.append(topk_logits.detach().cpu())

        # 4) Sample next token for each chain
        next_token_ids = _sample_from_logits(
            logits_next,
            temperature=sampling_cfg.temperature,
            top_k=sampling_cfg.top_k,
            top_p=sampling_cfg.top_p,
        )
        token_ids_steps.append(next_token_ids.detach().cpu())

        # 5) Append sampled tokens to each sequence (in python lists)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(num_chains):
            batch_token_ids[i].append(int(next_token_ids_list[i]))

    # -------------------------
    # Stack and convert to numpy
    # -------------------------

    # token_ids_steps: list of [N] → [T, N] → [N, T]
    token_ids_tensor = torch.stack(token_ids_steps, dim=0).permute(1, 0)  # [N, T]
    token_ids = token_ids_tensor.numpy().astype(np.int64)

    # embeddings_steps: list of [N, D] → [T, N, D] → [N, T, D]
    embeddings_tensor = torch.stack(embeddings_steps, dim=0).permute(1, 0, 2)  # [N, T, D]
    embeddings = embeddings_tensor.numpy().astype(np.float32)

    # top-k
    if sampling_cfg.store_topk_logits > 0 and len(topk_ids_steps) > 0:
        topk_ids_tensor = torch.stack(topk_ids_steps, dim=0).permute(1, 0, 2)       # [N, T, K]
        topk_logits_tensor = torch.stack(topk_logits_steps, dim=0).permute(1, 0, 2) # [N, T, K]
        topk_token_ids = topk_ids_tensor.numpy().astype(np.int64)
        topk_logits = topk_logits_tensor.numpy().astype(np.float32)
    else:
        topk_token_ids = None
        topk_logits = None

    # Simple step mask: everything is "real" (no early stopping yet)
    step_mask = np.ones_like(token_ids, dtype=np.int32)

    meta = {
        "model_name_or_path": model.config.model_name_or_path,
        "sampling_config": sampling_cfg,
    }

    batch = ChainBatch(
        prompt=prompt_spec,
        prompt_token_ids=prompt_token_ids_np,
        token_ids=token_ids,
        embeddings=embeddings,
        topk_token_ids=topk_token_ids,
        topk_logits=topk_logits,
        step_mask=step_mask,
        meta=meta,
    )
    return batch
