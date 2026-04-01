from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

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
    SpecAugment,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
)
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import squeezeformer_variant


def _checkpoint_name(epoch: int, val_wer: float) -> str:
    return f"checkpoint_epoch={epoch:04d}_valwer={val_wer:.6f}.pt"


class SchedulerDefaults(NamedTuple):
    peak_lr: float
    num_time_masks: int


class DTypeChoice(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


def _variant_defaults(variant: str) -> SchedulerDefaults:
    if variant in {"xs", "s", "sm"}:
        return SchedulerDefaults(peak_lr=2e-3, num_time_masks=5)
    if variant == "m":
        return SchedulerDefaults(peak_lr=1.5e-3, num_time_masks=7)
    return SchedulerDefaults(peak_lr=1e-3, num_time_masks=10)


def build_paper_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    warmup_epochs: int = 20,
    hold_epochs: int = 160,
    decay_exponent: float = 1.0,
):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    hold_steps = max(0, hold_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if current_step < warmup_steps + hold_steps:
            return 1.0
        decay_step = max(1, current_step - hold_steps)
        return (warmup_steps**decay_exponent) / (decay_step**decay_exponent)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_autocast_dtype(dtype: DTypeChoice) -> torch.dtype | None:
    if dtype == DTypeChoice.FLOAT32:
        return None
    if dtype == DTypeChoice.FLOAT16:
        return torch.float16
    if dtype == DTypeChoice.BFLOAT16:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _autocast_context(device: torch.device, dtype: DTypeChoice):
    autocast_dtype = _resolve_autocast_dtype(dtype)
    if autocast_dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "cpu" and autocast_dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on CPU. Use bfloat16 or float32.")
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


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
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=DTypeChoice,
        choices=list(DTypeChoice),
        default=DTypeChoice.BFLOAT16,
    )
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument(
        "--tokenizer",
        default="sentencepiece",
        choices=["character", "sentencepiece"],
    )
    parser.add_argument("--spm-vocab-size", type=int, default=128)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--hold-epochs", type=int, default=160)
    parser.add_argument("--decay-exponent", type=float, default=1.0)
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
    dtype: DTypeChoice,
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

            with _autocast_context(device, dtype):
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
    variant_defaults = _variant_defaults(args.variant)

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
    train_dataset = CV22ASRDataset(
        train_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        specaugment=SpecAugment(num_time_masks=variant_defaults.num_time_masks),
    )
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
    use_grad_scaler = device.type == "cuda" and args.dtype == DTypeChoice.FLOAT16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    peak_lr = args.learning_rate if args.learning_rate is not None else variant_defaults.peak_lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_paper_scheduler(
        optimizer,
        steps_per_epoch=max(1, len(train_loader)),
        warmup_epochs=args.warmup_epochs,
        hold_epochs=args.hold_epochs,
        decay_exponent=args.decay_exponent,
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

            with _autocast_context(device, args.dtype):
                log_probs, output_lengths = model.log_probs(features, feature_lengths)
                loss = criterion(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, tokenizer, device, args.dtype)
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
