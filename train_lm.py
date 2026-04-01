from __future__ import annotations

import argparse
import json
from pathlib import Path

from squeezeformer_pytorch.lm import NGramLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shallow-fusion character n-gram LM from a newline-delimited corpus."
    )
    parser.add_argument("--corpus", required=True, help="Path to a newline-delimited text corpus.")
    parser.add_argument(
        "--output",
        default="artifacts/shallow_fusion_lm.json",
        help="Path to the LM JSON artifact.",
    )
    parser.add_argument("--order", type=int, default=3, help="n-gram order.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Add-alpha smoothing value.")
    parser.add_argument(
        "--preview-text",
        default=None,
        help="Optional text to score after training.",
    )
    return parser.parse_args()


def read_corpus(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [line for line in lines if line]
    if not texts:
        raise ValueError(f"corpus '{path}' does not contain any non-empty lines")
    return texts


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = read_corpus(corpus_path)
    lm = NGramLanguageModel.train(texts, order=args.order, alpha=args.alpha)
    lm.save(output_path)

    summary: dict[str, object] = {
        "corpus": str(corpus_path),
        "output": str(output_path),
        "lines": len(texts),
        "order": lm.order,
        "alpha": lm.alpha,
        "vocabulary_size": len(lm.vocabulary),
        "lm_scorer_spec": f"squeezeformer_pytorch.lm:load_saved_ngram_scorer:{output_path}",
    }
    if args.preview_text:
        summary["preview_text"] = args.preview_text
        summary["preview_log_score"] = lm.score_text(args.preview_text)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
