from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    download_cv22_dataset,
    feature_cache_path,
    iter_cv22_records,
    load_audio,
    prevalidate_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache log-mel features for a dataset split without training."
    )
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Deterministic internal split to cache.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute features even if the cache file already exists.",
    )
    parser.add_argument(
        "--prevalidate-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prevalidate-workers", type=int, default=4)
    parser.add_argument("--min-transcript-chars", type=int, default=1)
    parser.add_argument("--max-transcript-chars", type=int, default=400)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--preemphasis", type=float, default=0.97)
    parser.add_argument(
        "--normalize-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-per-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    featurizer = AudioFeaturizer(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        preemphasis=args.preemphasis,
        normalize_signal=args.normalize_signal,
        normalize_feature=args.normalize_feature,
        normalize_per_frame=args.normalize_per_frame,
    )
    split_cache_dir = Path(args.feature_cache_dir) / args.split
    selection_progress = tqdm(
        desc=f"Selecting {args.split} records",
        total=args.max_samples,
        unit="utt",
    )

    def selected_records():
        try:
            for record in iter_cv22_records(
                dataset_root=dataset_root,
                split=args.split,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                max_samples=args.max_samples,
                min_transcript_chars=args.min_transcript_chars,
                max_transcript_chars=args.max_transcript_chars,
                max_symbol_ratio=args.max_symbol_ratio,
            ):
                selection_progress.update()
                yield record
        finally:
            selection_progress.close()

    records = selected_records()
    if args.prevalidate_audio:
        records = prevalidate_records(list(records), num_workers=args.prevalidate_workers)
        if not records:
            raise RuntimeError("Audio prevalidation removed every sample from the selected split.")

    counters = {"processed": 0, "written": 0, "cache_hits": 0}
    progress = tqdm(desc=f"Extracting {args.split} features", total=args.max_samples, unit="utt")
    try:
        for record in records:
            cache_path = feature_cache_path(split_cache_dir, record.utterance_id, featurizer)
            if cache_path is None:
                raise RuntimeError("feature cache path could not be resolved")
            counters["processed"] += 1
            progress.update()
            if cache_path.exists() and not args.overwrite:
                counters["cache_hits"] += 1
                continue
            waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
            features = featurizer(waveform, sample_rate)
            torch.save(features, cache_path)
            counters["written"] += 1
    finally:
        progress.close()

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "feature_cache_dir": str(split_cache_dir),
        "processed": counters["processed"],
        "written": counters["written"],
        "cache_hits": counters["cache_hits"],
        "overwrite": args.overwrite,
        "max_samples": args.max_samples,
        "featurizer_config": featurizer.config_dict(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
