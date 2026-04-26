from __future__ import annotations

import os
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

_PYPTX_DISABLED = os.environ.get("SQUEEZEFORMER_DISABLE_PYPTX", "").lower() in {
    "1",
    "true",
    "yes",
}
_BLOCK_CANDIDATES = (1024, 512, 256, 128, 64, 32)
_LOG2E = 1.4426950408889634
_PYPTX_IMPORT_READY = False


def _ensure_pyptx_importable() -> bool:
    global _PYPTX_IMPORT_READY
    if _PYPTX_DISABLED:
        return False
    if _PYPTX_IMPORT_READY:
        return True
    try:
        import_module("pyptx.kernel")
    except (ImportError, ModuleNotFoundError):
        vendored = Path(__file__).resolve().parents[1] / "pyptx"
        if not (vendored / "pyptx" / "__init__.py").exists():
            return False
        sys.path.insert(0, str(vendored))
        sys.modules.pop("pyptx", None)
        try:
            import_module("pyptx.kernel")
        except (ImportError, ModuleNotFoundError):
            return False
    _PYPTX_IMPORT_READY = True
    return True


def _is_inference_cuda_f32(x: Tensor, *others: Tensor) -> bool:
    if _PYPTX_DISABLED or torch.is_grad_enabled():
        return False
    tensors = (x, *others)
    if not all(
        tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        for tensor in tensors
    ):
        return False
    major, _minor = torch.cuda.get_device_capability(x.device)
    return major >= 9


def _pick_block(n: int) -> int:
    for block in _BLOCK_CANDIDATES:
        if n % (block * 4) == 0 and block >= 128:
            return block
    for block in _BLOCK_CANDIDATES:
        if n % block == 0:
            return block
    raise ValueError(f"feature dimension {n} is not supported by pyptx fast kernels")


def _flat_2d(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    return x.reshape(-1, original_shape[-1]), original_shape


@lru_cache(maxsize=128)
def _build_scale_bias_kernel(m: int, f: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, u32

    from pyptx import Tile, kernel, ptx, reg

    block = _pick_block(f)
    items_per_thread = f // block
    use_v4 = items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(m, f, f32), Tile(f, f32), Tile(f, f32)),
        out_specs=(Tile(m, f, f32),),
        grid=(m, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def scale_bias(x, scale, bias, out):
        px, ps, pb, po = ptx.global_ptrs(x, scale, bias, out)
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_byte_off = row * (f * 4)
        px += row_byte_off
        po += row_byte_off

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * 4
                x_vals = [reg.scalar(f32) for _ in range(4)]
                s_vals = [reg.scalar(f32) for _ in range(4)]
                b_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(x_vals, ptx.addr(px + off))
                ptx.inst.ld.global_.v4.f32(s_vals, ptx.addr(ps + off))
                ptx.inst.ld.global_.v4.f32(b_vals, ptx.addr(pb + off))
                y_vals = []
                for sub in range(4):
                    y = reg.scalar(f32)
                    ptx.inst.fma.rn.f32(y, x_vals[sub], s_vals[sub], b_vals[sub])
                    y_vals.append(y)
                ptx.inst.st.global_.v4.f32(ptx.addr(po + off), y_vals)
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                off = idx * 4
                xv = reg.scalar(f32)
                sv = reg.scalar(f32)
                bv = reg.scalar(f32)
                y = reg.scalar(f32)
                ptx.inst.ld.global_.f32(xv, ptx.addr(px + off))
                ptx.inst.ld.global_.f32(sv, ptx.addr(ps + off))
                ptx.inst.ld.global_.f32(bv, ptx.addr(pb + off))
                ptx.inst.fma.rn.f32(y, xv, sv, bv)
                ptx.inst.st.global_.f32(ptx.addr(po + off), y)
        ptx.ret()

    return scale_bias


@lru_cache(maxsize=128)
def _build_silu_kernel(m: int, f: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, u32

    from pyptx import Tile, kernel, ptx, reg

    block = _pick_block(f)
    items_per_thread = f // block
    use_v4 = items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(m, f, f32),),
        out_specs=(Tile(m, f, f32),),
        grid=(m, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def silu(x, out):
        px, po = ptx.global_ptrs(x, out)
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_byte_off = row * (f * 4)
        px += row_byte_off
        po += row_byte_off

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        neg_log2e = reg.scalar(f32, init=-_LOG2E)
        one = reg.scalar(f32, init=1.0)

        def emit_one(xv):
            neg_x = reg.scalar(f32)
            ptx.inst.mul.f32(neg_x, xv, neg_log2e)
            exp_neg = reg.scalar(f32)
            ptx.inst.ex2.approx.f32(exp_neg, neg_x)
            denom = reg.scalar(f32)
            ptx.inst.add.f32(denom, one, exp_neg)
            sigm = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(sigm, denom)
            y = reg.scalar(f32)
            ptx.inst.mul.f32(y, xv, sigm)
            return y

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * 4
                x_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(x_vals, ptx.addr(px + off))
                ptx.inst.st.global_.v4.f32(ptx.addr(po + off), [emit_one(v) for v in x_vals])
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                off = idx * 4
                xv = reg.scalar(f32)
                ptx.inst.ld.global_.f32(xv, ptx.addr(px + off))
                ptx.inst.st.global_.f32(ptx.addr(po + off), emit_one(xv))
        ptx.ret()

    return silu


def _arch_for(x: Tensor) -> str:
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major >= 10:
        return "sm_100a"
    return "sm_90a"


def scale_bias_or_torch(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    if not _is_inference_cuda_f32(x, scale, bias) or x.dim() < 1:
        return x * scale + bias
    flat, original_shape = _flat_2d(x)
    try:
        kernel = _build_scale_bias_kernel(flat.size(0), flat.size(1), _arch_for(x))
        return kernel(flat, scale, bias).reshape(original_shape)
    except Exception:
        return x * scale + bias


def silu_or_torch(x: Tensor) -> Tensor:
    if not _is_inference_cuda_f32(x) or x.dim() < 1:
        return F.silu(x)
    flat, original_shape = _flat_2d(x)
    try:
        kernel = _build_silu_kernel(flat.size(0), flat.size(1), _arch_for(x))
        return kernel(flat).reshape(original_shape)
    except Exception:
        return F.silu(x)
