from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import trackio
from torch import nn

from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    Tokenizer,
)
from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    CV22ASRDataset,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
)
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import squeezeformer_variant


def _checkpoint_name(epoch: int, val_wer: float) -> str:
    return f"checkpoint_epoch={epoch:04d}_valwer={val_wer:.6f}.pt"


def _update_top_checkpoints(
    output_dir: Path,
    checkpoint: dict[str, object],
    epoch: int,
    val_wer: float,
    keep_top_k: int,
) -> None:
    topk_dir = output_dir / "checkpoints_topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = topk_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = []

    filename = _checkpoint_name(epoch=epoch, val_wer=val_wer)
    checkpoint_path = topk_dir / filename
    torch.save(checkpoint, checkpoint_path)

    metadata.append(
        {
            "epoch": epoch,
            "val_wer": val_wer,
            "path": str(checkpoint_path.name),
        }
    )
    metadata.sort(key=lambda item: (float(item["val_wer"]), int(item["epoch"])))

    removed = metadata[keep_top_k:]
    metadata = metadata[:keep_top_k]
    for item in removed:
        stale_path = topk_dir / str(item["path"])
        if stale_path.exists():
            stale_path.unlink()

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Squeezeformer CTC on speech-uk/cv22.")
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--variant", default="sm", choices=["xs", "s", "sm", "m", "ml", "l"])
    parser.add_argument("--output-dir", default="artifacts/squeezeformer-cv22")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument(
        "--tokenizer",
        default="character",
        choices=["character", "sentencepiece"],
    )
    parser.add_argument("--spm-vocab-size", type=int, default=256)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    return parser.parse_args()


def greedy_decode(log_probs: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    return [tokenizer.decode_ctc(sequence) for sequence in token_ids]


def evaluate(
    model: SqueezeformerCTC,
    dataloader,
    criterion: nn.CTCLoss,
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
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
            total_batches += 1
            references.extend(batch["transcripts"])
            hypotheses.extend(greedy_decode(log_probs, tokenizer))

    return {
        "loss": total_loss / max(1, total_batches),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    train_records = load_cv22_records(
        dataset_root=dataset_root,
        split="train",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_train_samples,
    )
    val_records = load_cv22_records(
        dataset_root=dataset_root,
        split="validation",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_val_samples,
    )

    if args.tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer.train(
            (record.transcript for record in train_records),
            model_prefix=output_dir / "tokenizer",
            vocab_size=args.spm_vocab_size,
            model_type=args.spm_model_type,
        )
        tokenizer.save(output_dir / "tokenizer.model")
    else:
        tokenizer = CharacterTokenizer.build(record.transcript for record in train_records)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    featurizer = AudioFeaturizer()
    train_dataset = CV22ASRDataset(train_records, tokenizer=tokenizer, featurizer=featurizer)
    val_dataset = CV22ASRDataset(val_records, tokenizer=tokenizer, featurizer=featurizer)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encoder_config = squeezeformer_variant(args.variant)
    model = SqueezeformerCTC(encoder_config=encoder_config, vocab_size=tokenizer.vocab_size)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    trackio.init(
        project=args.trackio_project,
        space_id=args.trackio_space_id,
        config={
            **vars(args),
            "encoder_config": asdict(encoder_config),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
        },
    )

    best_val_wer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs, output_lengths = model.log_probs(features, feature_lengths)
            loss = criterion(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            if step % args.log_every == 0:
                trackio.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train_loss_step": float(loss.item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, tokenizer, device)
        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
        }
        trackio.log(log_payload)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "encoder_config": asdict(encoder_config),
            "tokenizer": tokenizer.to_dict(),
            "epoch": epoch,
            "metrics": log_payload,
        }
        latest_path = output_dir / "checkpoint_last.pt"
        torch.save(checkpoint, latest_path)
        if val_metrics["wer"] < best_val_wer:
            best_val_wer = val_metrics["wer"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        _update_top_checkpoints(
            output_dir=output_dir,
            checkpoint=checkpoint,
            epoch=epoch,
            val_wer=val_metrics["wer"],
            keep_top_k=args.keep_top_k,
        )

    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val_wer": best_val_wer,
                "variant": args.variant,
                "keep_top_k": args.keep_top_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trackio.finish()


if __name__ == "__main__":
    main()
