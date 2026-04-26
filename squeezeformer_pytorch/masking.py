from __future__ import annotations

import torch
from torch import Tensor


def make_sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    lengths = lengths.to(dtype=torch.long)
    if max_length is None:
        max_length = int(lengths.max().item()) if lengths.numel() else 0
    return torch.arange(int(max_length), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def make_padding_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    return ~make_sequence_mask(lengths, max_length=max_length)


def make_attention_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    sequence_mask = make_sequence_mask(lengths, max_length=max_length)
    return sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
