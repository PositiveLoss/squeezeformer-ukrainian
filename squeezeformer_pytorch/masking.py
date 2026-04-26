from __future__ import annotations

from torch import Tensor

from .pyptx_kernels import sequence_mask_or_torch, squeezeformer_attention_mask_or_torch


def make_sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    return sequence_mask_or_torch(lengths, max_length=max_length)


def make_padding_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    return ~make_sequence_mask(lengths, max_length=max_length)


def make_attention_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    return squeezeformer_attention_mask_or_torch(lengths, max_length=max_length)
