from __future__ import annotations

from argparse import Namespace

from hparam_tuner import build_train_command, estimate_training_hparams


def test_estimate_training_hparams_cpu_smoke() -> None:
    args = Namespace(
        variant="sm",
        optimizer="muon",
        tokenizer="sentencepiece",
        spm_vocab_size=128,
        device="cpu",
        dtype="bfloat16",
        feature_cache_dir="artifacts/feature_cache",
        speed_perturb_prob=0.5,
        noise_prob=0.2,
        reverb_prob=0.1,
        decode_strategy="beam",
        beam_size=8,
        output_dir="artifacts/cv22-sm",
        epochs=10,
        base_batch_size=8,
        base_max_batch_frames=12000,
        base_gradient_accumulation_steps=4,
        avg_frames_per_sample=1500,
        emit_format="json",
    )

    estimate = estimate_training_hparams(args)

    assert estimate.batch_size >= 1
    assert estimate.max_batch_frames >= 4000
    assert estimate.gradient_accumulation_steps >= 1
    assert estimate.num_workers >= 1
    assert estimate.metadata_workers >= 1
    assert estimate.prefetch_factor == 2
    assert estimate.beam_size <= 8
    assert estimate.estimated_effective_frames >= estimate.max_batch_frames


def test_build_train_command_includes_estimated_knobs() -> None:
    args = Namespace(
        variant="sm",
        optimizer="muon",
        tokenizer="sentencepiece",
        spm_vocab_size=128,
        device="cpu",
        dtype="bfloat16",
        feature_cache_dir="artifacts/feature_cache",
        speed_perturb_prob=0.5,
        noise_prob=0.2,
        reverb_prob=0.1,
        decode_strategy="beam",
        beam_size=8,
        output_dir="artifacts/cv22-sm",
        epochs=10,
        base_batch_size=8,
        base_max_batch_frames=12000,
        base_gradient_accumulation_steps=4,
        avg_frames_per_sample=1500,
        emit_format="shell",
    )
    estimate = estimate_training_hparams(args)

    command = build_train_command(args, estimate)

    assert "--device cpu" in command
    assert f"--batch-size {estimate.batch_size}" in command
    assert f"--max-batch-frames {estimate.max_batch_frames}" in command
    assert f"--gradient-accumulation-steps {estimate.gradient_accumulation_steps}" in command


def test_estimate_training_hparams_xla_smoke() -> None:
    args = Namespace(
        variant="sm",
        optimizer="muon",
        tokenizer="sentencepiece",
        spm_vocab_size=128,
        device="xla",
        dtype="bfloat16",
        feature_cache_dir="artifacts/feature_cache",
        speed_perturb_prob=0.5,
        noise_prob=0.2,
        reverb_prob=0.1,
        decode_strategy="beam",
        beam_size=8,
        output_dir="artifacts/cv22-sm",
        epochs=10,
        base_batch_size=8,
        base_max_batch_frames=12000,
        base_gradient_accumulation_steps=4,
        avg_frames_per_sample=1500,
        emit_format="json",
    )

    estimate = estimate_training_hparams(args)

    assert estimate.batch_size >= 1
    assert estimate.max_batch_frames >= 12000
    assert estimate.gradient_accumulation_steps >= 1
    assert estimate.num_workers >= 2
    assert estimate.metadata_workers >= 2
    assert estimate.prefetch_factor == 2
    assert estimate.beam_size <= 6
    assert estimate.estimated_effective_frames >= estimate.max_batch_frames
