from __future__ import annotations

import torch

from zipformer_pytorch.asr import (
    ZipformerConfig,
    ZipformerCTC,
    ZipformerEncoder,
)


def _tiny_zipformer_config() -> ZipformerConfig:
    return ZipformerConfig(
        input_dim=8,
        output_downsampling_factor=2,
        downsampling_factor=(1, 2),
        encoder_dim=(16, 24),
        num_encoder_layers=(1, 1),
        num_heads=(4, 4),
        query_head_dim=(8,),
        value_head_dim=(4,),
        pos_head_dim=(4,),
        feedforward_dim=(32, 48),
        cnn_module_kernel=(5, 5),
        pos_dim=16,
        dropout=0.0,
    )


def test_zipformer_ctc_returns_training_outputs_with_main_log_probs() -> None:
    model = ZipformerCTC(
        encoder_config=_tiny_zipformer_config(),
        vocab_size=6,
    )
    features = torch.randn(2, 12, 8)
    feature_lengths = torch.tensor([12, 9], dtype=torch.long)
    targets = torch.tensor([1, 2, 1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    outputs = model(
        features,
        feature_lengths,
        return_training_outputs=True,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=0,
        return_main_log_probs=True,
    )

    assert outputs["encoded"].shape == (2, 3, 24)
    assert outputs["output_lengths"].tolist() == [3, 3]
    assert outputs["main_log_probs"].shape == (2, 3, 6)
    assert outputs["main_ctc_loss"] is not None
    assert outputs["intermediate_ctc_losses"] == {}
    assert outputs["blank_logit_regularization_loss"].dtype == torch.float32


def test_zipformer_ctc_forward_runs_on_meta_device() -> None:
    model = ZipformerCTC(
        encoder_config=_tiny_zipformer_config(),
        vocab_size=6,
    ).to("meta")
    features = torch.randn(2, 12, 8, device="meta")
    feature_lengths = torch.tensor([12, 9], dtype=torch.long, device="meta")

    logits, output_lengths = model(features, feature_lengths)

    assert logits.device.type == "meta"
    assert output_lengths.device.type == "meta"


def test_zipformer_encoder_masks_padding_inside_blocks() -> None:
    torch.manual_seed(0)
    encoder = ZipformerEncoder(_tiny_zipformer_config()).eval()
    valid_length = torch.tensor([7], dtype=torch.long)
    base = torch.randn(1, 7, 8)
    short = torch.cat([base, torch.zeros(1, 2, 8)], dim=1)
    long = torch.cat([base, torch.zeros(1, 8, 8)], dim=1)

    with torch.no_grad():
        new_short, _ = encoder(short, valid_length)
        new_long, _ = encoder(long, valid_length)

    new_diff = (new_short[:, :2] - new_long[:, :2]).abs().max().item()

    assert new_diff < 0.01
