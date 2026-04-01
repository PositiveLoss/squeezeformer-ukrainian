from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def _register_legacy_main_aliases() -> None:
    """Expose train.py enums on the current __main__ module for legacy checkpoints."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    from train import AdaptiveBatchUnit, DTypeChoice, DecodeStrategy, OptimizerChoice

    for cls in (AdaptiveBatchUnit, DTypeChoice, DecodeStrategy, OptimizerChoice):
        if not hasattr(main_module, cls.__name__):
            setattr(main_module, cls.__name__, cls)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    _register_legacy_main_aliases()
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
