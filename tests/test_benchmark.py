from __future__ import annotations

from dataclasses import asdict

import benchmark
from squeezeformer_pytorch import squeezeformer_variant
from squeezeformer_pytorch.runtime_types import DTypeChoice


def test_load_model_uses_transformer_engine_for_fp8_checkpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32

    class DummyModel:
        def __init__(self, **kwargs: object) -> None:
            captured["use_transformer_engine"] = kwargs["use_transformer_engine"]

        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

    checkpoint = {
        "tokenizer": {"type": "character", "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {"dtype": "fp8"},
        "model_state_dict": {"weight": 1},
    }

    monkeypatch.setattr(benchmark, "load_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        benchmark,
        "tokenizer_from_dict",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(benchmark, "SqueezeformerCTC", DummyModel)

    args = type(
        "Args",
        (),
        {
            "checkpoint": "checkpoint.pt",
            "variant": "xs",
            "dtype": DTypeChoice.FLOAT32,
        },
    )()

    benchmark._load_model(args)

    assert captured["use_transformer_engine"] is True
    assert captured["state_dict"] == {"weight": 1}
