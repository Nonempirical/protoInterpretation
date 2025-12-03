from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFModelConfig:
    model_name_or_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", or "float32"


class HFModelAdapter:
    """
    Thin wrapper around a HuggingFace causal LM.

    Responsibilities:
      - Load model + tokenizer (from hub or local path).
      - Provide encode/decode helpers.
      - Given a batch of token ID sequences, return:
          * logits for next token
          * last-token hidden embedding
    """

    def __init__(self, config: HFModelConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Map dtype string to actual torch dtype
        if config.dtype == "float16":
            torch_dtype = torch.float16
        elif config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        # Ensure we have a pad token (important for batching)
        if self.tokenizer.pad_token is None:
            # fall back to eos token
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # last resort, add a pad token
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.pad_token_id = self.tokenizer.pad_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch_dtype,
        )

        # If we added new tokens, resize embeddings
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

        # For causal LM, left padding is usually safest
        self.tokenizer.padding_side = "left"

    # --------- constructors ----------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: str = "float16",
    ) -> "HFModelAdapter":
        """
        Convenience constructor.

        model_name_or_path:
            - HuggingFace hub name (e.g. "gpt2", "Qwen/xxx")
            - or local path (e.g. "/content/tmp_model_dir")
        """
        cfg = HFModelConfig(
            model_name_or_path=model_name_or_path,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=dtype,
        )
        return cls(cfg)

    # --------- basic helpers ----------

    def encode(self, text: str) -> List[int]:
        """Convert text → token IDs (without adding special BOS/EOS)."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs → text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.shape[0])

    # --------- internal padding helper ----------

    def _pad_batch(
        self,
        batch_token_ids: List[List[int]],
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Left-pad a batch of variable-length sequences.

        Returns:
          input_ids: [B, L]
          attention_mask: [B, L] (1 for real tokens, 0 for padding)
        """
        batch_size = len(batch_token_ids)
        max_len = max(len(seq) for seq in batch_token_ids)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for i, seq in enumerate(batch_token_ids):
            seq_len = len(seq)
            if seq_len == 0:
                continue
            # left pad: place sequence at the end
            input_ids[i, max_len - seq_len : max_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, max_len - seq_len : max_len] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return input_ids, attention_mask

    # --------- main method: logits + embeddings ----------

    @torch.no_grad()
    def get_logits_and_embeddings(
        self,
        batch_token_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of token ID sequences, returns:

        - next_token_logits: [B, V]    (logits for next token)
        - last_hidden:       [B, D]    (last token hidden state)

        The model is run once on the padded batch.
        """
        input_ids, attention_mask = self._pad_batch(batch_token_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Logits for all positions: [B, L, V]
        logits = outputs.logits
        # Hidden states for all layers: tuple(L_layers)[B, L, D]
        hidden_states = outputs.hidden_states

        # We care about the LAST token for each sequence.
        # Because of left padding, the last non-pad token is always at position -1.
        next_token_logits = logits[:, -1, :]          # [B, V]
        last_hidden = hidden_states[-1][:, -1, :]     # [B, D]

        return next_token_logits, last_hidden
