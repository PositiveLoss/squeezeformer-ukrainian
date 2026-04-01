# Squeezeformer PyTorch

This repository contains a standalone PyTorch implementation of the Squeezeformer encoder described in the paper source under [arXiv-2206.00888v2](/workspace/arXiv-2206.00888v2), plus minimal CTC training and evaluation scripts for the gated Hugging Face dataset `speech-uk/cv22`.

## What Is In The Repo

- [model.py](/workspace/squeezeformer_pytorch/model.py): Squeezeformer encoder
- [asr.py](/workspace/squeezeformer_pytorch/asr.py): CTC wrapper and tokenizer implementations
- [data.py](/workspace/squeezeformer_pytorch/data.py): dataset download, Polars manifest loading, audio featurization, batching
- [metrics.py](/workspace/squeezeformer_pytorch/metrics.py): CER and WER through `jiwer`
- [train.py](/workspace/train.py): training entrypoint
- [evaluate.py](/workspace/evaluate.py): evaluation entrypoint
- [tests](/workspace/tests): basic model and tokenizer checks

## Implemented Architecture

The encoder follows the paper’s main architectural changes:

- depthwise-separable 4x acoustic subsampling
- MF/CF block layout
- scaled post-LN residual modules
- relative-position attention
- Temporal U-Net style time reduction and recovery
- published size variants: `xs`, `s`, `sm`, `m`, `ml`, `l`

Published variant table used in code:

- `xs`: 16 layers, 144 dim, 4 heads
- `s`: 18 layers, 196 dim, 4 heads
- `sm`: 16 layers, 256 dim, 4 heads
- `m`: 20 layers, 324 dim, 4 heads
- `ml`: 18 layers, 512 dim, 8 heads
- `l`: 22 layers, 640 dim, 8 heads

This is still not a full paper reproduction, but the current code now aligns with several paper-critical defaults:

- greedy CTC decoding
- SentencePiece or character tokenization
- SentencePiece-128 as the default tokenizer configuration
- paper-style warmup, hold, and inverse-power decay scheduler
- variant-aware SpecAugment defaults

## Environment Setup

The workflow uses a local `uv` environment.

Create the environment:

```bash
uv venv .venv
```

Install dependencies:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install polars jiwer sentencepiece huggingface_hub trackio pytest ruff
```

## Dataset Access

The scripts use `speech-uk/cv22` from Hugging Face:

- dataset page: `https://huggingface.co/datasets/speech-uk/cv22`
- the dataset is gated
- you must accept the access conditions on Hugging Face before the scripts can download it

Set a token before training or evaluation:

```bash
export HF_TOKEN=your_huggingface_token
```

The loader downloads the dataset snapshot with `huggingface_hub.snapshot_download()` and reads TSV or Parquet manifests with `polars`.

Important: the dataset itself exposes only a train split. This repo creates deterministic internal `train`, `validation`, and `test` splits by hashing each utterance id with the provided seed and split fractions.

Default split behavior:

- `--val-fraction 0.1`
- `--test-fraction 0.1`

So the effective split is:

- 80% train
- 10% validation
- 10% test

This is record-based splitting, not speaker-aware splitting.

## Tokenization

Two tokenizer modes are supported:

- `character`
- `sentencepiece`

Character mode builds a vocabulary directly from the training transcripts.

SentencePiece mode trains a tokenizer during `train.py` and stores:

- `tokenizer.json`
- `tokenizer.model`

Evaluation reloads the tokenizer from checkpoint metadata, so you do not need to pass the tokenizer type again.

## Quick Start

Character tokenizer:

```bash
HF_TOKEN=... uv run python train.py \
  --variant sm \
  --tokenizer character \
  --device cpu \
  --output-dir artifacts/cv22-sm-char \
  --batch-size 8 \
  --epochs 10
```

SentencePiece tokenizer:

```bash
HF_TOKEN=... uv run python train.py \
  --variant sm \
  --tokenizer sentencepiece \
  --spm-vocab-size 128 \
  --spm-model-type unigram \
  --device cpu \
  --dtype bfloat16 \
  --output-dir artifacts/cv22-sm-spm \
  --batch-size 8 \
  --epochs 10
```

Evaluate:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm-spm/checkpoint_best.pt \
  --split test \
  --device cpu \
  --dtype bfloat16
```

## Training Script

[train.py](/workspace/train.py) does the following:

1. downloads the dataset snapshot
2. loads manifests with Polars
3. creates deterministic train/validation splits from the single source split
4. builds either a character tokenizer or a SentencePiece tokenizer from training transcripts
5. extracts 80-bin log-mel features with `torchaudio`
6. applies SpecAugment during training
7. trains a Squeezeformer encoder plus CTC head with the paper-style scheduler
7. logs metrics to `trackio`
8. saves checkpoints and tokenizer artifacts

Common arguments:

- `--dataset-repo`
- `--hf-token`
- `--cache-dir`
- `--variant`
- `--output-dir`
- `--batch-size`
- `--epochs`
- `--learning-rate`
- `--weight-decay`
- `--num-workers`
- `--seed`
- `--warmup-epochs`
- `--hold-epochs`
- `--decay-exponent`
- `--val-fraction`
- `--test-fraction`
- `--max-train-samples`
- `--max-val-samples`
- `--device`
- `--dtype`
- `--trackio-project`
- `--trackio-space-id`
- `--log-every`
- `--keep-top-k`
- `--tokenizer`
- `--spm-vocab-size`
- `--spm-model-type`

Important defaults:

- `--tokenizer sentencepiece`
- `--spm-vocab-size 128`
- `--spm-model-type unigram`
- `--dtype bfloat16`
- `--warmup-epochs 20`
- `--hold-epochs 160`
- `--decay-exponent 1.0`

Variant-aware defaults derived from the paper:

- `xs`, `s`, `sm`: peak LR `2e-3`, time masks `5`
- `m`: peak LR `1.5e-3`, time masks `7`
- `ml`, `l`: peak LR `1e-3`, time masks `10`

Files written by training:

- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `checkpoints_topk/` containing the best `--keep-top-k` checkpoints by validation WER
- `tokenizer.json`
- `tokenizer.model` when `--tokenizer sentencepiece`
- `train_summary.json`

Smoke-test example:

```bash
HF_TOKEN=... uv run python train.py \
  --variant xs \
  --device cpu \
  --output-dir artifacts/smoke \
  --epochs 1 \
  --max-train-samples 64 \
  --max-val-samples 16
```

## Evaluation Script

[evaluate.py](/workspace/evaluate.py) loads a checkpoint and computes:

- loss
- CER
- WER

It also logs the result to `trackio`.

Common arguments:

- `--checkpoint`
- `--dataset-repo`
- `--hf-token`
- `--cache-dir`
- `--split`
- `--batch-size`
- `--num-workers`
- `--seed`
- `--val-fraction`
- `--test-fraction`
- `--max-samples`
- `--device`
- `--dtype`
- `--trackio-project`
- `--trackio-space-id`

## Trackio

Training and evaluation use `trackio` for metric logging.

Local dashboard:

```bash
trackio show --project "squeezeformer-cv22"
```

To sync logs to a Hugging Face Space, pass:

- `--trackio-space-id username/space-name`

## Project Layout

```text
/workspace
├── README.md
├── pyproject.toml
├── train.py
├── evaluate.py
├── squeezeformer_pytorch
│   ├── __init__.py
│   ├── model.py
│   ├── asr.py
│   ├── data.py
│   └── metrics.py
├── tests
│   ├── conftest.py
│   └── test_squeezeformer.py
└── arXiv-2206.00888v2
```

## Validation

Current local checks:

```bash
uv run ruff check /workspace
uv run pytest -q
uv run python -c "import train, evaluate; print('imports_ok')"
```

## Known Limits

- This is still not the original large-scale TPU training environment from the paper.
- Decoding is greedy CTC only.
- The dataset loader is written for Common Voice-style manifests and should still be treated as a practical integration, not a benchmark-grade data pipeline.
- Splitting is record-based rather than speaker-aware.
- There is no beam search, language model fusion, mixed precision training, resume logic, or distributed training support.

## Sources

- Paper source: [arXiv-2206.00888v2](/workspace/arXiv-2206.00888v2)
- Dataset card: `https://huggingface.co/datasets/speech-uk/cv22`
- `trackio`: `https://pypi.org/project/trackio/`
