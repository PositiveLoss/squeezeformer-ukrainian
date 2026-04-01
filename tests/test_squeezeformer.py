import math
from pathlib import Path

import torch

from squeezeformer_pytorch import (
    SentencePieceTokenizer,
    SqueezeformerCTC,
    build_squeezeformer_encoder,
    squeezeformer_variant,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.data import SpecAugment
from train import OptimizerChoice, _variant_defaults, build_optimizer, build_paper_scheduler


def expected_subsampled_length(length: int) -> int:
    value = length
    for _ in range(2):
        value = math.floor(value / 2)
    return value


@torch.no_grad()
def test_variant_forward_shapes() -> None:
    batch = 2
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(batch, int(lengths.max().item()), 80)

    for variant in ["xs", "sm", "m"]:
        model = build_squeezeformer_encoder(variant)
        model.eval()
        outputs, output_lengths = model(features, lengths)

        cfg = squeezeformer_variant(variant)
        assert outputs.shape[0] == batch
        assert outputs.shape[2] == cfg.d_model
        expected_lengths = torch.tensor(
            [expected_subsampled_length(160), expected_subsampled_length(123)]
        )
        assert torch.equal(output_lengths, expected_lengths)
        assert outputs.shape[1] == int(output_lengths.max().item())
        assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_temporal_unet_recovers_subsampled_resolution() -> None:
    model = build_squeezeformer_encoder("sm")
    model.eval()
    lengths = torch.tensor([200], dtype=torch.int64)
    features = torch.randn(1, 200, 80)

    outputs, output_lengths = model(features, lengths)

    assert output_lengths.item() == expected_subsampled_length(200)
    assert outputs.shape[1] == output_lengths.item()


def test_variant_table_matches_paper() -> None:
    expected = {
        "xs": (16, 144, 4),
        "s": (18, 196, 4),
        "sm": (16, 256, 4),
        "m": (20, 324, 4),
        "ml": (18, 512, 8),
        "l": (22, 640, 8),
    }
    for name, values in expected.items():
        cfg = squeezeformer_variant(name)
        assert (cfg.num_layers, cfg.d_model, cfg.num_heads) == values


def test_sentencepiece_tokenizer_roundtrip(tmp_path: Path) -> None:
    tokenizer = SentencePieceTokenizer.train(
        ["привіт світе", "це короткий тест", "мовна модель"],
        model_prefix=tmp_path / "spm",
        vocab_size=24,
    )
    token_ids = tokenizer.encode("привіт світе")
    assert token_ids
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert restored.decode(token_ids)


def test_time_reduction_kernel_matches_paper() -> None:
    model = build_squeezeformer_encoder("sm")
    reduction = model.time_reduce["7"]
    assert reduction.kernel_size == 3


def test_paper_defaults_use_sentencepiece_compatible_variant_defaults() -> None:
    assert _variant_defaults("sm").peak_lr == 2e-3
    assert _variant_defaults("sm").num_time_masks == 5
    assert _variant_defaults("m").peak_lr == 1.5e-3
    assert _variant_defaults("m").num_time_masks == 7
    assert _variant_defaults("l").peak_lr == 1e-3
    assert _variant_defaults("l").num_time_masks == 10


def test_paper_scheduler_warmup_hold_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_paper_scheduler(
        optimizer,
        steps_per_epoch=2,
        warmup_epochs=1,
        hold_epochs=1,
        decay_exponent=1.0,
    )
    observed = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[0] == 1.0
    assert observed[1] == 1.0
    assert observed[2] == 1.0
    assert observed[3] < observed[2]
    assert observed[4] < observed[3]


def test_specaugment_preserves_shape() -> None:
    augment = SpecAugment(
        num_freq_masks=2,
        freq_mask_param=4,
        num_time_masks=2,
        time_mask_max_ratio=0.2,
    )
    features = torch.ones(20, 8)
    augmented = augment(features)
    assert augmented.shape == features.shape


def test_muon_optimizer_partition_uses_encoder_hidden_weights() -> None:
    config = squeezeformer_variant("xs")
    model = SqueezeformerCTC(encoder_config=config, vocab_size=32)
    optimizers, optimizer_names = build_optimizer(
        model=model,
        optimizer_name=OptimizerChoice.MUON,
        lr=1e-3,
        weight_decay=1e-4,
    )
    assert optimizer_names == ["muon", "adamw_aux"]
    muon_params = optimizers[0].param_groups[0]["params"]
    adamw_params = optimizers[1].param_groups[0]["params"]
    assert muon_params
    assert adamw_params
    assert all(parameter.ndim == 2 for parameter in muon_params)
