from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torchaudio
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .asr import CharacterTokenizer

TRANSCRIPT_COLUMNS = ("sentence", "transcript", "text", "normalized_text")
AUDIO_COLUMNS = ("path", "audio")


@dataclass(frozen=True)
class CV22Record:
    audio_path: str | None
    audio_bytes: bytes | None
    transcript: str
    utterance_id: str


class AudioFeaturizer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sample_rate,
            )[0]
        features = self.mel(waveform)
        return torch.log(features.clamp_min(1e-5)).transpose(0, 1)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _hash_to_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _collect_manifest_frames(dataset_root: Path) -> list[pl.LazyFrame]:
    manifest_frames: list[pl.LazyFrame] = []
    tsv_files = sorted(dataset_root.rglob("*.tsv"))
    if tsv_files:
        for path in tsv_files:
            manifest_frames.append(pl.scan_csv(path, separator="\t", infer_schema_length=1000))
        return manifest_frames

    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    for path in parquet_files:
        manifest_frames.append(pl.scan_parquet(path))
    return manifest_frames


def _extract_transcript(row: dict[str, Any]) -> str:
    for column in TRANSCRIPT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)
    raise KeyError(f"No transcript column found in row. Tried {TRANSCRIPT_COLUMNS}.")


def _resolve_audio(row: dict[str, Any], dataset_root: Path) -> tuple[str | None, bytes | None]:
    path_value = row.get("path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = dataset_root / path
        return str(path), None

    audio_value = row.get("audio")
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
        resolved_path: str | None = None
        if isinstance(audio_path, str) and audio_path:
            path = Path(audio_path)
            if not path.is_absolute():
                path = dataset_root / audio_path
            resolved_path = str(path)
        if isinstance(audio_bytes, (bytes, bytearray)):
            return resolved_path, bytes(audio_bytes)
        if resolved_path is not None:
            return resolved_path, None

    raise KeyError(f"No audio source found in row. Tried {AUDIO_COLUMNS}.")


def download_cv22_dataset(
    repo_id: str,
    token: str | None,
    cache_dir: str | None = None,
    force_download: bool = False,
) -> Path:
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        cache_dir=cache_dir,
        resume_download=not force_download,
    )
    return Path(local_path)


def load_cv22_records(
    dataset_root: Path,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
) -> list[CV22Record]:
    frames = _collect_manifest_frames(dataset_root)
    if not frames:
        raise FileNotFoundError(f"No TSV or Parquet manifest files found under {dataset_root}.")

    frame = pl.concat(frames, how="diagonal_relaxed").collect()
    records: list[CV22Record] = []
    for row in frame.iter_rows(named=True):
        try:
            transcript = _extract_transcript(row)
            audio_path, audio_bytes = _resolve_audio(row, dataset_root=dataset_root)
        except KeyError:
            continue
        utterance_id = str(row.get("client_id") or row.get("id") or audio_path or len(records))
        records.append(
            CV22Record(
                audio_path=audio_path,
                audio_bytes=audio_bytes,
                transcript=transcript,
                utterance_id=utterance_id,
            )
        )

    if not records:
        raise RuntimeError("No usable records were found in the dataset manifests.")

    selected: list[CV22Record] = []
    for record in records:
        score = _hash_to_unit_interval(record.utterance_id, seed=seed)
        train_cutoff = max(0.0, 1.0 - val_fraction - test_fraction)
        if split == "train" and score < train_cutoff:
            selected.append(record)
        elif split == "validation" and train_cutoff <= score < train_cutoff + val_fraction:
            selected.append(record)
        elif split == "test" and score >= train_cutoff + val_fraction:
            selected.append(record)

    if max_samples is not None:
        selected = selected[:max_samples]

    if not selected:
        raise RuntimeError(f"Split '{split}' is empty after applying the current split fractions.")
    return selected


class CV22ASRDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[CV22Record],
        tokenizer: CharacterTokenizer,
        featurizer: AudioFeaturizer,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.audio_path is not None and Path(record.audio_path).exists():
            waveform, sample_rate = torchaudio.load(record.audio_path)
        elif record.audio_bytes is not None:
            waveform, sample_rate = torchaudio.load(io.BytesIO(record.audio_bytes))
        else:
            raise FileNotFoundError(f"Audio for record {record.utterance_id} is not available.")

        features = self.featurizer(waveform, sample_rate)
        target_ids = torch.tensor(self.tokenizer.encode(record.transcript), dtype=torch.long)
        return {
            "features": features,
            "feature_length": features.size(0),
            "targets": target_ids,
            "target_length": target_ids.numel(),
            "transcript": record.transcript,
            "utterance_id": record.utterance_id,
        }


def collate_asr_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    feature_lengths = torch.tensor([item["feature_length"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    max_feature_length = int(feature_lengths.max().item())

    padded_features = []
    targets = []
    transcripts = []
    utterance_ids = []
    for item in batch:
        feature = item["features"]
        if feature.size(0) < max_feature_length:
            feature = F.pad(feature, (0, 0, 0, max_feature_length - feature.size(0)))
        padded_features.append(feature)
        targets.append(item["targets"])
        transcripts.append(item["transcript"])
        utterance_ids.append(item["utterance_id"])

    return {
        "features": torch.stack(padded_features, dim=0),
        "feature_lengths": feature_lengths,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "utterance_ids": utterance_ids,
    }


def create_dataloader(
    dataset: CV22ASRDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_asr_batch,
    )
