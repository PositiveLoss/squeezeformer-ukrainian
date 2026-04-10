from __future__ import annotations

from squeezeformer_pytorch.training.torchrun_launcher import (
    _parse_nproc_per_node,
    _prepare_torchrun_env,
    _recommended_omp_num_threads,
)


def test_parse_nproc_per_node_supports_both_torchrun_spellings() -> None:
    assert _parse_nproc_per_node(["--nproc_per_node=2", "train.py"]) == 2
    assert _parse_nproc_per_node(["--nproc-per-node", "4", "train.py"]) == 4
    assert _parse_nproc_per_node(["train.py"]) is None


def test_recommended_omp_num_threads_scales_with_local_world_size() -> None:
    assert _recommended_omp_num_threads(cpu_count=32, nproc_per_node=2) == 16
    assert _recommended_omp_num_threads(cpu_count=3, nproc_per_node=8) == 1
    assert _recommended_omp_num_threads(cpu_count=32, nproc_per_node=1) is None
    assert _recommended_omp_num_threads(cpu_count=None, nproc_per_node=2) == 1


def test_prepare_torchrun_env_sets_default_only_when_unset() -> None:
    launch_args = ["--nproc_per_node=2", "train.py"]
    prepared = _prepare_torchrun_env(
        launch_args,
        env={"PATH": "/tmp/bin"},
        cpu_count=24,
    )
    assert prepared["OMP_NUM_THREADS"] == "12"

    preserved = _prepare_torchrun_env(
        launch_args,
        env={"PATH": "/tmp/bin", "OMP_NUM_THREADS": "7"},
        cpu_count=24,
    )
    assert preserved["OMP_NUM_THREADS"] == "7"
