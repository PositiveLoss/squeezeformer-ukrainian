from __future__ import annotations

import torch
from torch import Tensor


def make_sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def make_padding_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    return ~make_sequence_mask(lengths, max_length=max_length)


def make_attention_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    sequence_mask = make_sequence_mask(lengths, max_length=max_length)
    return sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
