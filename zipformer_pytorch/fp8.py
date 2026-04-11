from __future__ import annotations

import math

import torch
from torch import Tensor, nn

try:
    import transformer_engine.pytorch as te
except (ImportError, OSError):
    te = None


FP8_SHAPE_ALIGNMENT = 16


def transformer_engine_available() -> bool:
    return te is not None


def _is_fp8_compatible_linear_shape(in_features: int, out_features: int) -> bool:
    return in_features % FP8_SHAPE_ALIGNMENT == 0 and out_features % FP8_SHAPE_ALIGNMENT == 0


def _padded_sequence_length(length: int) -> int:
    return math.ceil(length / FP8_SHAPE_ALIGNMENT) * FP8_SHAPE_ALIGNMENT


def _pad_tensor_along_dim(x: Tensor, dim: int, target_size: int) -> Tensor:
    if x.size(dim) >= target_size:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = target_size - x.size(dim)
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    use_transformer_engine: bool = False,
) -> nn.Module:
    if (
        use_transformer_engine
        and te is not None
        and _is_fp8_compatible_linear_shape(in_features, out_features)
    ):
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def apply_linear_with_fp8_padding(module: nn.Module, x: Tensor) -> Tensor:
    if te is None or not isinstance(module, te.Linear) or x.dim() < 2:
        return module(x)

    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    padded_rows = _padded_sequence_length(flat.size(0))
    if padded_rows != flat.size(0):
        flat = _pad_tensor_along_dim(flat, dim=0, target_size=padded_rows)
    flat = module(flat)
    flat = flat[: math.prod(original_shape[:-1])]
    return flat.reshape(*original_shape[:-1], flat.size(-1))
