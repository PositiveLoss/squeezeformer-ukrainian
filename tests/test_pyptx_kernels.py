from __future__ import annotations

import importlib


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
