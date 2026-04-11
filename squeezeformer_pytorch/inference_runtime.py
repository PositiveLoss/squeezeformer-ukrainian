from __future__ import annotations

import re
from typing import Any


def resolve_inference_checkpoint_settings(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    training_args = checkpoint_data.get("training_args", {})

    return {
        "aed_decoder_enabled": bool(training_args.get("aed_decoder", False)),
        "aed_decoder_layers": int(training_args.get("aed_decoder_layers", 1)),
        "aed_decoder_heads": int(training_args.get("aed_decoder_heads", 4)),
        "aed_decoder_dropout": float(training_args.get("aed_decoder_dropout", 0.1)),
        "liberta_distill_enabled": bool(training_args.get("liberta_distill", False)),
        "audio_teacher_enabled": bool(training_args.get("audio_teacher", False)),
        "audio_teacher_target": str(training_args.get("audio_teacher_target", "encoder")),
    }


def normalize_transcript_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_transcripts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith((" ", "\n")) or right.startswith((" ", "\n", ",", ".", "!", "?", ":", ";")):
        return f"{left}{right}"
    return f"{left} {right}"


def merge_chunk_transcript(existing: str, new: str) -> str:
    existing = normalize_transcript_whitespace(existing)
    new = normalize_transcript_whitespace(new)
    if not existing:
        return new
    if not new:
        return existing
    if new in existing:
        return existing
    if existing in new:
        return new

    existing_words = existing.split(" ")
    new_words = new.split(" ")
    max_word_overlap = min(len(existing_words), len(new_words), 20)
    for overlap in range(max_word_overlap, 0, -1):
        if existing_words[-overlap:] == new_words[:overlap]:
            return " ".join(existing_words + new_words[overlap:])

    max_char_overlap = min(len(existing), len(new), 120)
    for overlap in range(max_char_overlap, 8, -1):
        if existing[-overlap:] == new[:overlap]:
            return f"{existing}{new[overlap:]}"

    return join_transcripts(existing, new)
