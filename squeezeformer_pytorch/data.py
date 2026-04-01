from __future__ import annotations

import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torchaudio
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .asr import Tokenizer

TRANSCRIPT_COLUMNS = ("sentence", "transcript", "text", "normalized_text")
AUDIO_COLUMNS = ("path", "audio")


@dataclass(frozen=True)
class CV22Record:
    audio_path: str | None
    audio_bytes: bytes | None
    transcript: str
    utterance_id: str
    speaker_id: str
    estimated_frames: int


class AudioFeaturizer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        preemphasis: float = 0.97,
        normalize_signal: bool = True,
        normalize_feature: bool = True,
        normalize_per_frame: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature
        self.normalize_per_frame = normalize_per_frame
        self.hop_length = hop_length
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
        if self.normalize_signal:
            waveform = waveform - waveform.mean()
            waveform = waveform / waveform.abs().amax().clamp_min(1e-6)
        if self.preemphasis > 0:
            waveform = torch.cat(
                [waveform[:1], waveform[1:] - self.preemphasis * waveform[:-1]],
                dim=0,
            )
        features = self.mel(waveform)
        features = torch.log(features.clamp_min(1e-5)).transpose(0, 1)
        if self.normalize_feature:
            if self.normalize_per_frame:
                mean = features.mean(dim=-1, keepdim=True)
                std = features.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            else:
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
            features = (features - mean) / std
        return features

    def config_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "preemphasis": self.preemphasis,
            "normalize_signal": self.normalize_signal,
            "normalize_feature": self.normalize_feature,
            "normalize_per_frame": self.normalize_per_frame,
            "hop_length": self.hop_length,
        }


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_mask_param: int = 27,
        num_time_masks: int = 5,
        time_mask_max_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_max_ratio = time_mask_max_ratio

    def forward(self, features: Tensor) -> Tensor:
        augmented = features.clone()
        time_steps, feature_bins = augmented.shape

        for _ in range(self.num_freq_masks):
            width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if width == 0 or width >= feature_bins:
                continue
            start = int(torch.randint(0, feature_bins - width + 1, (1,)).item())
            augmented[:, start : start + width] = 0

        max_time_width = max(1, int(time_steps * self.time_mask_max_ratio))
        for _ in range(self.num_time_masks):
            width = int(torch.randint(0, max_time_width + 1, (1,)).item())
            if width == 0 or width >= time_steps:
                continue
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            augmented[start : start + width, :] = 0

        return augmented


def normalize_transcript(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    normalized = re.sub(r"[“”«»]", '"', normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


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
            return normalize_transcript(value)
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
        utterance_id = str(row.get("id") or audio_path or len(records))
        speaker_id = str(row.get("client_id") or row.get("speaker_id") or utterance_id)
        duration_seconds = (
            row.get("duration") or row.get("duration_seconds") or row.get("audio_duration")
        )
        if isinstance(duration_seconds, (float, int)):
            estimated_frames = max(1, int((float(duration_seconds) * 16000) / 160))
        else:
            estimated_frames = 0
        records.append(
            CV22Record(
                audio_path=audio_path,
                audio_bytes=audio_bytes,
                transcript=transcript,
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                estimated_frames=estimated_frames,
            )
        )

    if not records:
        raise RuntimeError("No usable records were found in the dataset manifests.")

    selected: list[CV22Record] = []
    for record in records:
        score = _hash_to_unit_interval(record.speaker_id, seed=seed)
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
        tokenizer: Tokenizer,
        featurizer: AudioFeaturizer,
        specaugment: SpecAugment | None = None,
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.specaugment = specaugment
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def _feature_cache_path(self, record: CV22Record) -> Path | None:
        if self.feature_cache_dir is None:
            return None
        frontend_hash = hashlib.sha256(
            repr(self.featurizer.config_dict()).encode("utf-8")
        ).hexdigest()[:12]
        return self.feature_cache_dir / f"{record.utterance_id}_{frontend_hash}.pt"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cache_path = self._feature_cache_path(record)
        if cache_path is not None and cache_path.exists():
            features = torch.load(cache_path, map_location="cpu")
        else:
            if record.audio_path is not None and Path(record.audio_path).exists():
                waveform, sample_rate = torchaudio.load(record.audio_path)
            elif record.audio_bytes is not None:
                waveform, sample_rate = torchaudio.load(io.BytesIO(record.audio_bytes))
            else:
                raise FileNotFoundError(f"Audio for record {record.utterance_id} is not available.")
            features = self.featurizer(waveform, sample_rate)
            if cache_path is not None:
                torch.save(features, cache_path)
        if self.specaugment is not None:
            features = self.specaugment(features)
        target_ids = torch.tensor(self.tokenizer.encode(record.transcript), dtype=torch.long)
        return {
            "features": features,
            "feature_length": features.size(0),
            "targets": target_ids,
            "target_length": target_ids.numel(),
            "transcript": record.transcript,
            "utterance_id": record.utterance_id,
        }


class LengthBucketBatchSampler(BatchSampler):
    def __init__(self, records: list[CV22Record], batch_size: int, shuffle: bool) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = sorted(
            range(len(self.records)), key=lambda index: self.records[index].estimated_frames
        )
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


def _record_is_valid(record: CV22Record) -> bool:
    try:
        if record.audio_path is not None and Path(record.audio_path).exists():
            torchaudio.info(record.audio_path)
            return True
        if record.audio_bytes is not None:
            torchaudio.info(io.BytesIO(record.audio_bytes))
            return True
    except Exception:
        return False
    return False


def prevalidate_records(
    records: list[CV22Record],
    num_workers: int = 4,
) -> list[CV22Record]:
    if num_workers <= 1:
        return [record for record in records if _record_is_valid(record)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        validity = list(executor.map(_record_is_valid, records))
    return [record for record, is_valid in zip(records, validity, strict=True) if is_valid]


def estimate_record_frames(record: CV22Record, hop_length: int) -> int:
    if record.estimated_frames > 0:
        return record.estimated_frames
    if record.audio_path is not None and Path(record.audio_path).exists():
        try:
            info = torchaudio.info(record.audio_path)
            return max(1, int(info.num_frames / hop_length))
        except Exception:
            return 0
    return 0


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
    bucket_by_length: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, Any]]:
    for record in dataset.records:
        if record.estimated_frames <= 0:
            object.__setattr__(
                record,
                "estimated_frames",
                estimate_record_frames(record, hop_length=dataset.featurizer.hop_length),
            )
    dataloader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_asr_batch,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    if bucket_by_length:
        batch_sampler = LengthBucketBatchSampler(
            dataset.records, batch_size=batch_size, shuffle=shuffle
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )
