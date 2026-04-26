from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import squeezeformer_pytorch.pyptx_kernels as pyptx_kernels  # noqa: E402
from squeezeformer_pytorch.pyptx_kernels import (  # noqa: E402
    bias_norm_or_torch,
    conv_output_epilogue_or_torch,
    ctc_log_prob_frame_stats_or_torch,
    gated_linear_unit_bdt_or_torch,
    gated_linear_unit_or_torch,
    layer_norm_silu_scale_or_torch,
    scale_bias_or_torch,
    silu_time_mask_or_torch,
    swoosh_l_or_torch,
    swoosh_r_or_torch,
)


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[torch.device], tuple[Callable[[], Any], Callable[[], Any]]]


VARIANT_DIMS = {
    "xs": 144,
    "s": 196,
    "sm": 256,
    "m": 324,
    "ml": 512,
    "l": 640,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pyptx helper kernels against PyTorch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default="sm", choices=sorted(VARIANT_DIMS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="Floating point dtype for tensor-valued benchmark cases.",
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Wrap both helper and Torch reference callables in torch.compile before timing.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Configure Python logging verbosity.",
    )
    parser.add_argument(
        "--log-pyptx",
        action="store_true",
        help="Enable DEBUG logging for squeezeformer_pytorch.pyptx_kernels.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["all"],
        help="Case names to run, or 'all'.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def pyptx_disabled():
    previous = pyptx_kernels._PYPTX_DISABLED
    pyptx_kernels._PYPTX_DISABLED = True
    try:
        yield
    finally:
        pyptx_kernels._PYPTX_DISABLED = previous


@contextmanager
def capture_pyptx_fallbacks():
    records: list[logging.LogRecord] = []

    class _FallbackHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "fallback" in record.getMessage():
                records.append(record)

    logger = logging.getLogger(pyptx_kernels.__name__)
    handler = _FallbackHandler(level=logging.DEBUG)
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(min(previous_level, logging.DEBUG) if previous_level else logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def time_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup_iters: int,
    iters: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup_iters):
            out = fn()
            if isinstance(out, torch.Tensor):
                out.sum().item() if out.numel() == 1 and device.type != "cuda" else None
        synchronize(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            return float(start.elapsed_time(end) / iters)

        import time

        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        return 1000.0 * (time.perf_counter() - start_time) / iters


def lengths_for(batch: int, time: int, device: torch.device) -> torch.Tensor:
    if batch == 1:
        return torch.tensor([time], device=device, dtype=torch.long)
    values = torch.linspace(time, max(1, time // 2), steps=batch, device=device)
    return values.round().to(dtype=torch.long).clamp_(1, time)


def benchmark_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def make_cases(args: argparse.Namespace) -> list[Case]:
    batch = args.batch_size
    time = args.time_steps
    dim = VARIANT_DIMS[args.variant]
    hidden_dim = dim * 4
    conv_dim = dim * 2
    vocab = args.vocab_size
    dtype = benchmark_dtype(args.dtype)

    def randn(*shape: int, device: torch.device) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype)

    def scale_bias_case(device: torch.device):
        x = randn(batch, time, dim, device=device)
        scale = randn(dim, device=device)
        bias = randn(dim, device=device)
        return (
            lambda: scale_bias_or_torch(x, scale, bias),
            lambda: x * scale + bias,
        )

    def silu_mask_bdt_case(device: torch.device):
        x = randn(batch, conv_dim, time, device=device)
        mask = lengths_for(batch, time, device).unsqueeze(1) > torch.arange(time, device=device)
        return (
            lambda: silu_time_mask_or_torch(x, mask, layout="bdt"),
            lambda: F.silu(x) * mask.unsqueeze(1).to(dtype=x.dtype),
        )

    def silu_mask_btd_case(device: torch.device):
        x = randn(batch, time, dim, device=device)
        mask = lengths_for(batch, time, device).unsqueeze(1) > torch.arange(time, device=device)
        return (
            lambda: silu_time_mask_or_torch(x, mask, layout="btd"),
            lambda: F.silu(x) * mask.unsqueeze(-1).to(dtype=x.dtype),
        )

    def swoosh_l_case(device: torch.device):
        x = randn(batch, time, hidden_dim, device=device)
        return (
            lambda: swoosh_l_or_torch(x),
            lambda: F.softplus(x - 4.0) - 0.08 * x - 0.035,
        )

    def swoosh_r_case(device: torch.device):
        x = randn(batch, time, conv_dim, device=device)
        return (
            lambda: swoosh_r_or_torch(x),
            lambda: F.softplus(x - 1.0) - 0.08 * x - 0.313261687,
        )

    def gated_linear_unit_case(device: torch.device):
        x = randn(batch, time, conv_dim, device=device)
        return (
            lambda: gated_linear_unit_or_torch(x),
            lambda: x[..., :dim] * torch.sigmoid(x[..., dim:]),
        )

    def gated_linear_unit_bdt_case(device: torch.device):
        x = randn(batch, conv_dim, time, device=device)
        return (
            lambda: gated_linear_unit_bdt_or_torch(x),
            lambda: x[:, :dim, :] * torch.sigmoid(x[:, dim:, :]),
        )

    def conv_output_epilogue_case(device: torch.device):
        residual = randn(batch, time, dim, device=device)
        x = randn(batch, dim, time, device=device)
        mask = lengths_for(batch, time, device).unsqueeze(1) > torch.arange(time, device=device)
        return (
            lambda: conv_output_epilogue_or_torch(residual, x, mask),
            lambda: residual + (x * mask.unsqueeze(1).to(dtype=x.dtype)).transpose(1, 2),
        )

    def bias_norm_case(device: torch.device):
        x = randn(batch, time, dim, device=device)
        bias = randn(dim, device=device)
        log_scale = torch.randn((), device=device, dtype=dtype) * 0.1
        return (
            lambda: bias_norm_or_torch(x, bias, log_scale, 1.0e-8),
            lambda: (
                x
                / (x - bias.view(1, 1, dim)).pow(2).mean(dim=-1, keepdim=True).add(1.0e-8).sqrt()
                * log_scale.exp()
            ),
        )

    def layer_norm_silu_scale_case(device: torch.device):
        x = randn(batch, time, dim, device=device)
        weight = randn(dim, device=device)
        bias = randn(dim, device=device)
        confidence = torch.rand(batch, time, device=device, dtype=dtype)
        eps = 1.0e-5
        return (
            lambda: layer_norm_silu_scale_or_torch(x, weight, bias, confidence, eps),
            lambda: (
                F.silu(F.layer_norm(x, (dim,), weight, bias, eps))
                * confidence.unsqueeze(-1).clamp_min(0.1)
            ),
        )

    def ctc_stats_case(device: torch.device):
        logits = torch.randn(batch, time, vocab, device=device)
        log_probs = F.log_softmax(logits, dim=-1)
        lengths = lengths_for(batch, time, device)
        return (
            lambda: ctc_log_prob_frame_stats_or_torch(log_probs, lengths, blank_id=0),
            lambda: _torch_ctc_frame_stats(log_probs, lengths, blank_id=0),
        )

    return [
        Case("scale_bias", scale_bias_case),
        Case("silu_mask_bdt", silu_mask_bdt_case),
        Case("silu_mask_btd", silu_mask_btd_case),
        Case("swoosh_l", swoosh_l_case),
        Case("swoosh_r", swoosh_r_case),
        Case("gated_linear_unit", gated_linear_unit_case),
        Case("gated_linear_unit_bdt", gated_linear_unit_bdt_case),
        Case("conv_output_epilogue", conv_output_epilogue_case),
        Case("bias_norm", bias_norm_case),
        Case("layer_norm_silu_scale", layer_norm_silu_scale_case),
        Case("ctc_log_prob_frame_stats", ctc_stats_case),
    ]


def _torch_ctc_frame_stats(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    *,
    blank_id: int,
) -> torch.Tensor:
    valid = (
        torch.arange(log_probs.size(1), device=output_lengths.device).unsqueeze(0)
        < output_lengths.unsqueeze(1)
    ).to(dtype=torch.float32)
    nonblank = log_probs.clone()
    nonblank[..., blank_id] = float("-inf")
    return torch.stack(
        (
            log_probs[..., blank_id].exp() * valid,
            valid,
            log_probs.argmax(dim=-1).eq(blank_id).to(dtype=torch.float32) * valid,
            nonblank.max(dim=-1).values.exp() * valid,
        ),
        dim=-1,
    )


def max_abs_diff(a: Any, b: Any) -> float:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return 0.0
    if a.dtype == torch.bool or b.dtype == torch.bool:
        return float(a.ne(b).sum().item())
    return float((a.float() - b.float()).abs().max().item())


def compile_callable(fn: Callable[[], Any]) -> Callable[[], Any]:
    return torch.compile(fn, fullgraph=False, dynamic=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s"
    )
    if args.log_pyptx:
        logging.getLogger(pyptx_kernels.__name__).setLevel(logging.DEBUG)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    os.environ["SQUEEZEFORMER_DISABLE_PYPTX"] = "0"
    pyptx_kernels._PYPTX_DISABLED = False
    selected = set(args.cases)
    cases = make_cases(args)
    if selected != {"all"}:
        cases = [case for case in cases if case.name in selected]
    if not cases:
        raise ValueError(f"No benchmark cases selected from {sorted(selected)}.")

    results = []
    for case in cases:
        pyptx_fn, torch_fn = case.build(device)
        if args.torch_compile:
            pyptx_fn = compile_callable(pyptx_fn)
            torch_fn = compile_callable(torch_fn)

        with torch.inference_mode():
            with capture_pyptx_fallbacks() as fallback_records:
                pyptx_out = pyptx_fn()
            with pyptx_disabled():
                torch_out = torch_fn()
            synchronize(device)
        diff = max_abs_diff(pyptx_out, torch_out)
        pyptx_fell_back = bool(fallback_records)

        pyptx_ms = time_callable(
            pyptx_fn,
            device=device,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )
        with pyptx_disabled():
            torch_ms = time_callable(
                torch_fn,
                device=device,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        speedup = torch_ms / pyptx_ms if pyptx_ms > 0 else float("inf")
        results.append(
            {
                "case": case.name,
                "mode": ("compile" if args.torch_compile else "pyptx")
                if device.type == "cuda"
                else ("compile" if args.torch_compile else "fallback"),
                "pyptx_fallback": pyptx_fell_back,
                "pyptx_ms": pyptx_ms,
                "torch_ms": torch_ms,
                "speedup": speedup,
                "max_abs_diff": diff,
            }
        )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    name_width = max(len("case"), *(len(row["case"]) for row in results))
    print(
        f"{'case':<{name_width}}  {'mode':>8}  {'fallback':>8}  {'helper_ms':>10}  {'torch_ms':>10}  {'speedup':>8}  {'max_diff':>10}"
    )
    for row in results:
        print(
            f"{row['case']:<{name_width}}  {row['mode']:>8}  {str(row['pyptx_fallback']):>8}  "
            f"{row['pyptx_ms']:10.4f}  {row['torch_ms']:10.4f}  {row['speedup']:8.3f}  "
            f"{row['max_abs_diff']:10.3g}"
        )


if __name__ == "__main__":
    main()
