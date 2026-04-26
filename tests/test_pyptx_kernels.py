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


def test_fused_conv_output_epilogue_matches_torch_fallback() -> None:
    from squeezeformer_pytorch.pyptx_kernels import conv_output_epilogue_or_torch

    residual = torch.randn(2, 3, 4)
    x_bdt = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    expected = residual + (x_bdt * mask.unsqueeze(1).to(dtype=x_bdt.dtype)).transpose(1, 2)

    assert torch.allclose(conv_output_epilogue_or_torch(residual, x_bdt, mask), expected)


def test_arch_for_uses_sm120_for_rtx_blackwell(monkeypatch) -> None:
    import squeezeformer_pytorch.pyptx_kernels as pyptx_kernels

    class _FakeTensor:
        device = torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))

    assert pyptx_kernels._arch_for(_FakeTensor()) == "sm_120"
