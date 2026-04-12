from __future__ import annotations

import sys

from feature_cache_warmer.cli import (
    FeatureCacheWarmDataset,
    _resolve_cache_warm_splits,
    main,
    parse_args,
)

__all__ = [
    "FeatureCacheWarmDataset",
    "_resolve_cache_warm_splits",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main(sys.argv[1:])
