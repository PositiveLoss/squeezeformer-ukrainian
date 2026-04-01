from __future__ import annotations

import base64
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
from torch import Tensor, nn
from torch.nn import functional as F

from .model import SqueezeformerConfig, SqueezeformerEncoder


class Tokenizer:
    blank_id: int
    pad_id: int

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: Iterable[int]) -> str:
        raise NotImplementedError

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    def __init__(self, symbols: list[str]) -> None:
        if "<blank>" in symbols:
            raise ValueError("Do not include the reserved <blank> token in symbols.")
        self.blank_id = 0
        self.pad_id = 0
        self.id_to_token = ["<blank>"] + symbols
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1) -> "CharacterTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        symbols = sorted(token for token, count in counter.items() if count >= min_frequency)
        return cls(symbols=symbols)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[index] for index in token_ids if index != self.blank_id)

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        result: list[str] = []
        previous = self.blank_id
        for token_id in token_ids:
            if token_id != self.blank_id and token_id != previous:
                result.append(self.id_to_token[token_id])
            previous = token_id
        return "".join(result)

    def to_dict(self) -> dict[str, object]:
        return {"type": "character", "symbols": self.id_to_token[1:]}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharacterTokenizer":
        symbols = payload.get("symbols")
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ValueError("Tokenizer payload must contain a string list under 'symbols'.")
        return cls(symbols=symbols)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, processor: spm.SentencePieceProcessor, model_proto: bytes) -> None:
        self.processor = processor
        self.model_proto = model_proto
        self.blank_id = 0
        self.pad_id = 0

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_prefix: str | Path,
        vocab_size: int = 256,
        model_type: str = "unigram",
        character_coverage: float = 1.0,
    ) -> "SentencePieceTokenizer":
        model_prefix = str(model_prefix)
        text_path = f"{model_prefix}.training.txt"
        with open(text_path, "w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(text.replace("\n", " ").strip())
                handle.write("\n")

        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
        )
        model_path = f"{model_prefix}.model"
        model_proto = Path(model_path).read_bytes()
        processor = spm.SentencePieceProcessor(model_file=model_path)
        return cls(processor=processor, model_proto=model_proto)

    @classmethod
    def from_model_bytes(cls, model_proto: bytes) -> "SentencePieceTokenizer":
        processor = spm.SentencePieceProcessor(model_proto=model_proto)
        return cls(processor=processor, model_proto=model_proto)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, token_ids: Iterable[int]) -> str:
        filtered = [token_id for token_id in token_ids if token_id != self.blank_id]
        return self.processor.decode(filtered)

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        result: list[int] = []
        previous = self.blank_id
        for token_id in token_ids:
            if token_id != self.blank_id and token_id != previous:
                result.append(token_id)
            previous = token_id
        return self.processor.decode(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "sentencepiece",
            "model_proto_b64": base64.b64encode(self.model_proto).decode("ascii"),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".model":
            path.write_bytes(self.model_proto)
            return
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def tokenizer_from_dict(payload: dict[str, object]) -> Tokenizer:
    tokenizer_type = payload.get("type", "character")
    if tokenizer_type == "character":
        return CharacterTokenizer.from_dict(payload)
    if tokenizer_type == "sentencepiece":
        model_b64 = payload.get("model_proto_b64")
        if not isinstance(model_b64, str):
            raise ValueError("SentencePiece tokenizer payload must contain 'model_proto_b64'.")
        return SentencePieceTokenizer.from_model_bytes(base64.b64decode(model_b64))
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> Tokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tokenizer_from_dict(payload)


class SqueezeformerCTC(nn.Module):
    def __init__(self, encoder_config: SqueezeformerConfig, vocab_size: int) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.encoder = SqueezeformerEncoder(encoder_config)
        self.classifier = nn.Linear(encoder_config.d_model, vocab_size)

    def forward(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = self.classifier(encoded)
        return logits, output_lengths

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        logits, output_lengths = self(features, feature_lengths)
        return F.log_softmax(logits, dim=-1), output_lengths

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)
