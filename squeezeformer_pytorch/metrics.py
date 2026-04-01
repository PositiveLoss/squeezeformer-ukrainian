from __future__ import annotations

from typing import Iterable


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def char_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_edits = 0
    total_chars = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        total_edits += _edit_distance(list(reference), list(hypothesis))
        total_chars += max(1, len(reference))
    return total_edits / max(1, total_chars)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        total_edits += _edit_distance(reference_words, hypothesis_words)
        total_words += max(1, len(reference_words))
    return total_edits / max(1, total_words)
