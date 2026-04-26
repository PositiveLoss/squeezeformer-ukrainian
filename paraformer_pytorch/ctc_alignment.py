from __future__ import annotations

import torch

from ._ctc_alignment_python import (
    batch_ctc_viterbi_alignments_python,
    batch_uniform_alignments,
    ctc_viterbi_alignment_python,
)

try:
    from ._ctc_alignment_cython import (
        batch_ctc_viterbi_alignments_cython,
        ctc_viterbi_alignment_cython,
    )

    CYTHON_BACKEND_AVAILABLE = True
except ImportError:
    batch_ctc_viterbi_alignments_cython = None
    ctc_viterbi_alignment_cython = None
    CYTHON_BACKEND_AVAILABLE = False

try:
    from ._ctc_alignment_cuda import batch_ctc_viterbi_alignments_cuda

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    batch_ctc_viterbi_alignments_cuda = None
    CUDA_BACKEND_AVAILABLE = False

try:
    from ._ctc_alignment_triton import batch_ctc_viterbi_alignments_triton

    TRITON_BACKEND_AVAILABLE = True
except ImportError:
    batch_ctc_viterbi_alignments_triton = None
    TRITON_BACKEND_AVAILABLE = False


def _resolve_backend(backend: str, device: torch.device) -> str:
    if backend == "python":
        return "python"
    if backend == "triton":
        if device.type != "cuda":
            raise RuntimeError("Triton CTC alignment backend requires CUDA tensors.")
        if not TRITON_BACKEND_AVAILABLE:
            raise RuntimeError(
                "Triton CTC alignment backend is not importable. Install the `triton` dependency."
            )
        return "triton"
    if backend == "cython":
        if device.type == "cuda":
            if CUDA_BACKEND_AVAILABLE:
                return "cuda"
            if CYTHON_BACKEND_AVAILABLE:
                return "cython"
            raise RuntimeError(
                "Neither CUDA nor CPU Cython CTC alignment backends are built. "
                "Run `python setup.py build_ext --inplace`."
            )
        if not CYTHON_BACKEND_AVAILABLE:
            raise RuntimeError(
                "Cython CTC alignment backend is not built. Run `python setup.py build_ext --inplace`."
            )
        return "cython"
    if backend == "auto":
        if device.type == "cuda" and TRITON_BACKEND_AVAILABLE:
            return "triton"
        if device.type == "cuda" and CUDA_BACKEND_AVAILABLE:
            return "cuda"
        if CYTHON_BACKEND_AVAILABLE:
            return "cython"
        return "python"
    raise ValueError(f"unknown backend: {backend}")


def _to_cpu_float32(array: torch.Tensor) -> torch.Tensor:
    return array.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _to_cpu_long(array: torch.Tensor) -> torch.Tensor:
    return array.detach().to(device="cpu", dtype=torch.long).contiguous()


def ctc_viterbi_alignment(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    backend: str = "auto",
) -> torch.Tensor:
    resolved = _resolve_backend(backend, log_probs.device)
    if resolved == "python":
        return ctc_viterbi_alignment_python(log_probs, targets, blank_id)
    if resolved == "triton":
        result = batch_ctc_viterbi_alignments_triton(
            log_probs.detach().to(dtype=torch.float32).contiguous().unsqueeze(0),
            torch.tensor([log_probs.size(0)], dtype=torch.long, device=log_probs.device),
            targets.detach().to(dtype=torch.long, device=log_probs.device).contiguous().unsqueeze(0),
            torch.tensor([targets.numel()], dtype=torch.long, device=log_probs.device),
            int(blank_id),
        )
        return result[0, : log_probs.size(0)]
    if resolved == "cuda":
        result = batch_ctc_viterbi_alignments_cuda(
            log_probs.detach().to(dtype=torch.float32).contiguous().unsqueeze(0),
            torch.tensor([log_probs.size(0)], dtype=torch.long, device=log_probs.device),
            targets.detach().to(dtype=torch.long, device=log_probs.device).contiguous().unsqueeze(0),
            torch.tensor([targets.numel()], dtype=torch.long, device=log_probs.device),
            int(blank_id),
        )
        return result[0, : log_probs.size(0)]

    result = ctc_viterbi_alignment_cython(
        _to_cpu_float32(log_probs).numpy(),
        _to_cpu_long(targets).numpy(),
        int(blank_id),
    )
    return torch.from_numpy(result).to(device=log_probs.device)


def batch_ctc_viterbi_alignments(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    backend: str = "auto",
) -> torch.Tensor:
    resolved = _resolve_backend(backend, log_probs.device)
    if resolved == "python":
        return batch_ctc_viterbi_alignments_python(
            log_probs,
            input_lengths,
            targets,
            target_lengths,
            blank_id,
        )
    if resolved == "triton":
        return batch_ctc_viterbi_alignments_triton(
            log_probs.detach().to(dtype=torch.float32).contiguous(),
            input_lengths.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            targets.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            target_lengths.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            int(blank_id),
        )
    if resolved == "cuda":
        return batch_ctc_viterbi_alignments_cuda(
            log_probs.detach().to(dtype=torch.float32).contiguous(),
            input_lengths.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            targets.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            target_lengths.detach().to(dtype=torch.long, device=log_probs.device).contiguous(),
            int(blank_id),
        )

    result = batch_ctc_viterbi_alignments_cython(
        _to_cpu_float32(log_probs).numpy(),
        _to_cpu_long(input_lengths).numpy(),
        _to_cpu_long(targets).numpy(),
        _to_cpu_long(target_lengths).numpy(),
        int(blank_id),
    )
    return torch.from_numpy(result).to(device=log_probs.device)


__all__ = [
    "CYTHON_BACKEND_AVAILABLE",
    "CUDA_BACKEND_AVAILABLE",
    "TRITON_BACKEND_AVAILABLE",
    "batch_ctc_viterbi_alignments",
    "batch_ctc_viterbi_alignments_python",
    "batch_uniform_alignments",
    "ctc_viterbi_alignment",
    "ctc_viterbi_alignment_python",
]
