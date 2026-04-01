import math

import torch

from squeezeformer_pytorch import build_squeezeformer_encoder, squeezeformer_variant


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
