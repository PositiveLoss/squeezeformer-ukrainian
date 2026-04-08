from __future__ import annotations

from squeezeformer_pytorch.training.data_loading import _shard_records_for_rank


def test_shard_records_for_rank_allow_uneven_preserves_tail_samples() -> None:
    records = list(range(10))

    rank_0 = _shard_records_for_rank(records, rank=0, world_size=3, allow_uneven=True)
    rank_1 = _shard_records_for_rank(records, rank=1, world_size=3, allow_uneven=True)
    rank_2 = _shard_records_for_rank(records, rank=2, world_size=3, allow_uneven=True)

    assert rank_0 == [0, 3, 6, 9]
    assert rank_1 == [1, 4, 7]
    assert rank_2 == [2, 5, 8]


def test_shard_records_for_rank_even_mode_drops_tail_samples() -> None:
    records = list(range(10))

    rank_0 = _shard_records_for_rank(records, rank=0, world_size=3)
    rank_1 = _shard_records_for_rank(records, rank=1, world_size=3)
    rank_2 = _shard_records_for_rank(records, rank=2, world_size=3)

    assert rank_0 == [0, 3, 6]
    assert rank_1 == [1, 4, 7]
    assert rank_2 == [2, 5, 8]
