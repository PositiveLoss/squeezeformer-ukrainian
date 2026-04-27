#!/usr/bin/env python3
"""Extract model weights from PositiveLoss checkpoints.

The output is a plain f32 safetensors file plus a JSON sidecar. It keeps the
original module keys so downstream loaders can use architecture-specific name
aliases without losing information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeezeformer_pytorch.checkpoints import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract architecture weights from a PositiveLoss checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help=".pt/.pth/.ckpt or .safetensors checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive model.safetensors and weights.json",
    )
    parser.add_argument(
        "--architecture",
        choices=["auto", "squeezeformer", "zipformer", "paraformer", "w2v-bert"],
        default="auto",
        help="Override architecture metadata when it cannot be inferred.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Optional state_dict prefix to strip. Can be repeated.",
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Keep optimizer/scheduler metadata in the JSON sidecar when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_any_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    state_dict = normalize_state_dict(state_dict, strip_prefixes=args.strip_prefix)
    architecture = infer_architecture(checkpoint, state_dict, args.architecture)

    tensors = {
        key: tensor.detach().cpu().contiguous().to(torch.float32)
        for key, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }
    if not tensors:
        raise SystemExit(f"No tensor weights found in {checkpoint_path}")

    weights_path = output_dir / "model.safetensors"
    save_safetensors(
        tensors,
        str(weights_path),
        metadata={
            "format": "positiveloss-weights",
            "architecture": architecture,
            "source": str(checkpoint_path),
        },
    )

    metadata = {
        "format": "positiveloss-weights",
        "architecture": architecture,
        "source": str(checkpoint_path),
        "weights": str(weights_path),
        "tensor_count": len(tensors),
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "encoder_config": checkpoint.get("encoder_config") if isinstance(checkpoint, Mapping) else None,
        "training_args": checkpoint.get("training_args") if isinstance(checkpoint, Mapping) else None,
    }
    if args.keep_optimizer and isinstance(checkpoint, Mapping):
        for key in ("epoch", "global_step", "best_val_wer"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]

    metadata_path = output_dir / "weights.json"
    metadata_path.write_text(
        json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(tensors)} tensor(s) for {architecture} to {weights_path}")
    print(f"wrote metadata to {metadata_path}")


def load_any_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            return load_checkpoint(path, map_location="cpu", metadata_path=sidecar)
        return {"model_state_dict": load_safetensors(str(path), device="cpu")}
    return load_checkpoint(path, map_location="cpu")


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("resume_model_state_dict", "model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and any(torch.is_tensor(item) for item in value.values()):
            return dict(value)
    if any(torch.is_tensor(item) for item in checkpoint.values()):
        return dict(checkpoint)  # raw state_dict
    raise KeyError("checkpoint does not contain a model_state_dict-like mapping")


def normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strip_prefixes: list[str],
) -> dict[str, torch.Tensor]:
    prefixes = ["_orig_mod.", *strip_prefixes]
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if prefix and new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def infer_architecture(
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    encoder_config = checkpoint.get("encoder_config")
    if isinstance(encoder_config, Mapping):
        architecture = str(encoder_config.get("architecture", "")).replace("_", "-")
        if architecture:
            if architecture.startswith("paraformer"):
                return "paraformer"
            if architecture in {"squeezeformer", "zipformer", "w2v-bert"}:
                return architecture

    keys = tuple(state_dict.keys())
    if any(key.startswith("model.ctc_projection.") for key in keys):
        return "paraformer"
    if any(key.startswith("encoder.encoder.") for key in keys):
        return "zipformer"
    if any("wav2vec" in key or "w2v" in key for key in keys):
        return "w2v-bert"
    return "squeezeformer"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return value


if __name__ == "__main__":
    main()
