from __future__ import annotations

from typing import Any


def resolve_evaluation_checkpoint_settings(checkpoint: dict[str, Any]) -> dict[str, Any]:
    training_args = checkpoint.get("training_args", {})

    return {
        "aed_decoder_enabled": bool(training_args.get("aed_decoder", False)),
        "aed_decoder_layers": int(training_args.get("aed_decoder_layers", 1)),
        "aed_decoder_heads": int(training_args.get("aed_decoder_heads", 4)),
        "aed_decoder_dropout": float(training_args.get("aed_decoder_dropout", 0.1)),
        "aed_loss_weight": float(training_args.get("aed_loss_weight", 0.3)),
        "liberta_distill_enabled": bool(training_args.get("liberta_distill", False)),
        "liberta_model_name": str(training_args.get("liberta_model_name", "Yehor/Liberta")),
        "liberta_distill_weight": float(training_args.get("liberta_distill_weight", 0.05)),
        "liberta_max_length": int(training_args.get("liberta_max_length", 256)),
        "audio_teacher_enabled": bool(training_args.get("audio_teacher", False)),
        "audio_teacher_model_name": str(
            training_args.get("audio_teacher_model_name", "facebook/wav2vec2-bert-2.0")
        ),
        "audio_teacher_weight": float(training_args.get("audio_teacher_weight", 0.05)),
        "audio_teacher_objective": str(training_args.get("audio_teacher_objective", "hidden_mse")),
        "audio_teacher_target": str(training_args.get("audio_teacher_target", "encoder")),
        "audio_teacher_layer": int(training_args.get("audio_teacher_layer", -1)),
        "audio_teacher_sample_rate": int(training_args.get("audio_teacher_sample_rate", 16000)),
        "audio_teacher_max_seconds": float(training_args.get("audio_teacher_max_seconds", 30.0)),
    }


def resolve_lowercase_transcripts(
    override: bool | None,
    *,
    checkpoint_tokenizer_type: str,
) -> bool:
    if override is not None:
        return override
    return checkpoint_tokenizer_type != "sentencepiece"


def build_evaluation_payload(
    *,
    metrics: dict[str, object],
    result: dict[str, object],
    split: str,
    samples: int,
    decode_strategy: object,
) -> dict[str, object]:
    enriched_metrics = metrics | {
        "split": split,
        "samples": samples,
        "decode_strategy": decode_strategy,
    }
    return {
        "metrics": enriched_metrics,
        "hardest_examples": result["hardest_examples"],
        "random_examples": result["random_examples"],
        "speaker_metrics": result["speaker_metrics"],
    }
