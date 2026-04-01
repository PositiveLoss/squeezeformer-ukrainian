from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class NGramLanguageModel:
    order: int
    alpha: float
    vocabulary: list[str]
    context_totals: dict[str, int]
    next_token_counts: dict[str, dict[str, int]]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        order: int = 3,
        alpha: float = 0.1,
    ) -> "NGramLanguageModel":
        if order < 1:
            raise ValueError("order must be at least 1")

        vocabulary = set()
        context_totals: Counter[str] = Counter()
        next_token_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

        for text in texts:
            normalized = text.strip()
            if not normalized:
                continue
            vocabulary.update(normalized)
            history = ["<s>"] * max(0, order - 1) + list(normalized) + ["</s>"]
            for index in range(max(0, order - 1), len(history)):
                context = tuple(history[max(0, index - order + 1) : index])
                token = history[index]
                context_key = cls._context_key(context)
                context_totals[context_key] += 1
                next_token_counts[context_key][token] += 1

        if not vocabulary:
            raise ValueError("cannot train an n-gram LM on empty text")

        return cls(
            order=order,
            alpha=alpha,
            vocabulary=sorted(vocabulary) + ["</s>"],
            context_totals=dict(context_totals),
            next_token_counts={
                context: dict(counter)
                for context, counter in next_token_counts.items()
            },
        )

    @staticmethod
    def _context_key(context: tuple[str, ...]) -> str:
        return "\u241f".join(context)

    def score_extension(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = list(text)
        context = tuple((["<s>"] * max(0, self.order - 1) + tokens[:-1])[-max(0, self.order - 1) :])
        return self._score_token(context, tokens[-1])

    def score_text(self, text: str, include_eos: bool = True) -> float:
        total = 0.0
        history = ["<s>"] * max(0, self.order - 1)
        for token in text:
            context = tuple(history[-max(0, self.order - 1) :])
            total += self._score_token(context, token)
            history.append(token)
        if include_eos:
            context = tuple(history[-max(0, self.order - 1) :])
            total += self._score_token(context, "</s>")
        return total

    def _score_token(self, context: tuple[str, ...], token: str) -> float:
        context_key = self._context_key(context)
        counts = self.next_token_counts.get(context_key, {})
        numerator = counts.get(token, 0) + self.alpha
        denominator = self.context_totals.get(context_key, 0) + self.alpha * len(self.vocabulary)
        return math.log(numerator / denominator)

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "vocabulary": self.vocabulary,
            "context_totals": self.context_totals,
            "next_token_counts": self.next_token_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NGramLanguageModel":
        vocabulary = payload.get("vocabulary")
        context_totals = payload.get("context_totals")
        next_token_counts = payload.get("next_token_counts")
        if not isinstance(vocabulary, list) or not all(
            isinstance(item, str) for item in vocabulary
        ):
            raise ValueError("LM payload must contain a string list under 'vocabulary'")
        if not isinstance(context_totals, dict) or not isinstance(next_token_counts, dict):
            raise ValueError("LM payload must contain context and transition count dictionaries")
        return cls(
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
            vocabulary=vocabulary,
            context_totals={str(key): int(value) for key, value in context_totals.items()},
            next_token_counts={
                str(context): {str(token): int(count) for token, count in counts.items()}
                for context, counts in next_token_counts.items()
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NGramLanguageModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_saved_ngram_scorer(path: str) -> callable:
    lm = NGramLanguageModel.load(path)
    return lm.score_extension
