from __future__ import annotations

import torch

from paraformer_pytorch.asr import ParaformerASR
from paraformer_pytorch.model import ParaformerV2Config


def test_paraformer_asr_training_contract_smoke() -> None:
    config = ParaformerV2Config(
        input_dim=8,
        vocab_size=6,
        blank_id=0,
        encoder_dim=16,
        decoder_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ff_dim=32,
        decoder_ff_dim=32,
        attention_heads=2,
    )
    model = ParaformerASR(config, alignment_mode="uniform")
    features = torch.randn(2, 24, 8)
    feature_lengths = torch.tensor([24, 20])
    targets = torch.tensor([[1, 2, 3], [2, 4, 0]])
    target_lengths = torch.tensor([3, 2])

    outputs = model(
        features,
        feature_lengths,
        return_training_outputs=True,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=0,
        return_main_log_probs=True,
    )

    assert outputs["main_ctc_loss"].ndim == 0
    assert outputs["main_log_probs"].shape[:2] == (2, 6)
    assert outputs["main_log_probs"].shape[-1] == config.vocab_size
    assert outputs["output_lengths"].tolist() == [6, 5]
    outputs["main_ctc_loss"].backward()
