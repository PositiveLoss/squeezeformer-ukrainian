from __future__ import annotations

import importlib

import torch
from torch.nn import functional as F


def test_pyptx_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SQUEEZEFORMER_DISABLE_PYPTX", raising=False)

    import squeezeformer_pytorch.pyptx_kernels as pyptx_kernels

    reloaded = importlib.reload(pyptx_kernels)
    assert reloaded._PYPTX_DISABLED is True


def test_pyptx_can_be_enabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("SQUEEZEFORMER_DISABLE_PYPTX", "0")

    import squeezeformer_pytorch.pyptx_kernels as pyptx_kernels

    reloaded = importlib.reload(pyptx_kernels)
    assert reloaded._PYPTX_DISABLED is False


def test_fused_silu_time_mask_matches_torch_fallback() -> None:
    from squeezeformer_pytorch.pyptx_kernels import silu_time_mask_or_torch

    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    assert torch.allclose(
        silu_time_mask_or_torch(x, mask, layout="btd"),
        F.silu(x) * mask.unsqueeze(-1).to(dtype=x.dtype),
    )

    x_bdt = x.transpose(1, 2).contiguous()
    assert torch.allclose(
        silu_time_mask_or_torch(x_bdt, mask, layout="bdt"),
        F.silu(x_bdt) * mask.unsqueeze(1).to(dtype=x_bdt.dtype),
    )


def test_fused_residual_add_matches_torch_fallback() -> None:
    from squeezeformer_pytorch.pyptx_kernels import residual_add_or_torch

    residual = torch.randn(2, 3, 4)
    x = torch.randn(2, 3, 4)

    assert torch.allclose(residual_add_or_torch(residual, x, 0.5), residual + 0.5 * x)
