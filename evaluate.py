from __future__ import annotations

import argparse
import json
import os

import torch
import trackio
from torch import nn

from squeezeformer_pytorch.asr import SqueezeformerCTC, Tokenizer, tokenizer_from_dict
from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    CV22ASRDataset,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
)
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import SqueezeformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Squeezeformer CTC model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    return parser.parse_args()


def greedy_decode(log_probs: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    return [tokenizer.decode_ctc(sequence) for sequence in token_ids]


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    encoder_config = SqueezeformerConfig(**checkpoint["encoder_config"])
    model = SqueezeformerCTC(encoder_config=encoder_config, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    records = load_cv22_records(
        dataset_root=dataset_root,
        split=args.split,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_samples,
    )
    dataset = CV22ASRDataset(records, tokenizer=tokenizer, featurizer=AudioFeaturizer())
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    total_loss = 0.0
    references: list[str] = []
    hypotheses: list[str] = []
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            log_probs, output_lengths = model.log_probs(features, feature_lengths)
            loss = criterion(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            total_loss += float(loss.item())
            references.extend(batch["transcripts"])
            hypotheses.extend(greedy_decode(log_probs, tokenizer))

    metrics = {
        "split": args.split,
        "loss": total_loss / max(1, len(dataloader)),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
        "samples": len(records),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    trackio.init(
        project=args.trackio_project,
        space_id=args.trackio_space_id,
        config={"evaluation_split": args.split, "checkpoint": str(args.checkpoint)},
    )
    trackio.log(metrics)
    trackio.finish()


if __name__ == "__main__":
    main()
