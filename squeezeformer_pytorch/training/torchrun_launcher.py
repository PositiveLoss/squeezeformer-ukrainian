from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence


def _parse_positive_int(value: str, *, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{option} must be greater than zero, got {parsed}.")
    return parsed


def _parse_nproc_per_node(args: Sequence[str]) -> int | None:
    for index, arg in enumerate(args):
        if arg.startswith("--nproc_per_node="):
            return _parse_positive_int(arg.split("=", 1)[1], option="--nproc_per_node")
        if arg.startswith("--nproc-per-node="):
            return _parse_positive_int(arg.split("=", 1)[1], option="--nproc-per-node")
        if arg in {"--nproc_per_node", "--nproc-per-node"}:
            if index + 1 >= len(args):
                raise ValueError(f"{arg} requires a value.")
            return _parse_positive_int(args[index + 1], option=arg)
    return None


def _recommended_omp_num_threads(
    *,
    cpu_count: int | None,
    nproc_per_node: int | None,
) -> int | None:
    if nproc_per_node is None or nproc_per_node <= 1:
        return None
    available_cpus = cpu_count if cpu_count is not None and cpu_count > 0 else 1
    return max(1, available_cpus // nproc_per_node)


def _prepare_torchrun_env(
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cpu_count: int | None = None,
) -> dict[str, str]:
    prepared_env = dict(os.environ if env is None else env)
    if prepared_env.get("OMP_NUM_THREADS"):
        return prepared_env
    nproc_per_node = _parse_nproc_per_node(args)
    recommended = _recommended_omp_num_threads(
        cpu_count=cpu_count if cpu_count is not None else os.cpu_count(),
        nproc_per_node=nproc_per_node,
    )
    if recommended is not None:
        prepared_env["OMP_NUM_THREADS"] = str(recommended)
    return prepared_env


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    env = _prepare_torchrun_env(args)
    injected_omp_num_threads = "OMP_NUM_THREADS" in env and "OMP_NUM_THREADS" not in os.environ
    if injected_omp_num_threads:
        nproc_per_node = _parse_nproc_per_node(args)
        print(
            "Setting OMP_NUM_THREADS=%s for torchrun (cpu_count=%s nproc_per_node=%s). "
            "Export OMP_NUM_THREADS before launch to override this default."
            % (
                env["OMP_NUM_THREADS"],
                os.cpu_count() if os.cpu_count() is not None else "unknown",
                nproc_per_node,
            ),
            file=sys.stderr,
        )
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", *args],
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
