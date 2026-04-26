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


def _is_cuda_int64(x: Tensor) -> bool:
    if _PYPTX_DISABLED:
        return False
    if not (x.is_cuda and x.dtype == torch.int64 and x.is_contiguous()):
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
def _build_attention_mask_kernel(batch: int, time: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import pred, s64, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    items_per_thread = (time + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, s64),),
        out_specs=(Tile(batch, time, s64),),
        grid=(batch, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def attention_mask(lengths, out):
        pl, po = ptx.global_ptrs(lengths, out)

        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())

        length = reg.scalar(s64)
        ptx.inst.ld.global_.s64(length, ptx.addr(pl + row * 8))
        out_row = po + row * (time * 8)

        one = reg.scalar(s64, init=1)
        zero = reg.scalar(s64, init=0)
        for i in range(items_per_thread):
            col = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, col, time)
            col_s64 = reg.scalar(s64)
            ptx.inst.cvt.s64.u32(col_s64, col)
            is_valid = reg.scalar(pred)
            ptx.inst.setp.lt.s64(is_valid, col_s64, length)
            value = reg.scalar(s64)
            ptx.selp(s64, value, one, zero, is_valid)
            ptx.inst.st.global_.s64(ptx.addr(out_row + col * 8), value, pred=in_bounds)
        ptx.ret()

    return attention_mask


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
def _build_masked_mean_kernel(batch: int, time: int, dim: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, s64, u32

    from pyptx import Tile, kernel, ptx, reg, smem

    warp_size = 32
    block = 256
    num_warps = block // warp_size
    items_per_thread = (time + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, time, dim, f32), Tile(batch, time, s64)),
        out_specs=(Tile(batch, dim, f32),),
        grid=(batch, dim, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def masked_mean(hidden, mask, out):
        partials = smem.alloc(f32, (num_warps, 2))
        stats = smem.alloc(f32, (1, 1))
        ph, pm, po = ptx.global_ptrs(hidden, mask, out)

        batch_idx = reg.scalar(u32)
        ptx.inst.mov.u32(batch_idx, ptx.special.ctaid.x())
        feature_idx = reg.scalar(u32)
        ptx.inst.mov.u32(feature_idx, ptx.special.ctaid.y())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & (warp_size - 1)
        warp_id = tid >> 5

        hidden_row_base = batch_idx * (time * dim * 4)
        mask_row_base = batch_idx * (time * 8)
        sum_value = reg.scalar(f32, init=0.0)
        count_value = reg.scalar(f32, init=0.0)
        zero_s64 = reg.scalar(s64, init=0)
        one_f32 = reg.scalar(f32, init=1.0)

        for i in range(items_per_thread):
            col = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, col, time)

            mask_value = reg.scalar(s64, init=0)
            ptx.inst.ld.global_.s64(
                mask_value,
                ptx.addr(pm + mask_row_base + col * 8),
                pred=in_bounds,
            )
            is_valid = reg.scalar(pred)
            ptx.inst.setp.ne.s64(is_valid, mask_value, zero_s64)

            hidden_value = reg.scalar(f32, init=0.0)
            hidden_off = hidden_row_base + (col * dim + feature_idx) * 4
            ptx.inst.ld.global_.f32(
                hidden_value,
                ptx.addr(ph + hidden_off),
                pred=is_valid,
            )
            ptx.inst.add.f32(sum_value, sum_value, hidden_value)
            ptx.inst.add.f32(count_value, count_value, one_f32, pred=is_valid)

        ptx.warp.reduce_sum(sum_value)
        ptx.warp.reduce_sum(count_value)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = sum_value
            partials[warp_id, 1] = count_value
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            block_count = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
                ptx.inst.add.f32(block_count, block_count, partials[i, 1])

            has_count = reg.scalar(pred)
            ptx.inst.setp.gt.f32(has_count, block_count, 0.0)
            denom = reg.scalar(f32)
            ptx.selp(f32, denom, block_count, one_f32, has_count)
            inv_denom = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_denom, denom)
            pooled = reg.scalar(f32)
            ptx.inst.mul.f32(pooled, block_sum, inv_denom)
            stats[0, 0] = pooled
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            ptx.inst.st.global_.f32(
                ptx.addr(po + (batch_idx * dim + feature_idx) * 4),
                stats[0, 0],
            )
        ptx.ret()

    return masked_mean


@lru_cache(maxsize=128)
def _build_layer_norm_kernel(rows: int, dim: int, eps: float, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, u32

    from pyptx import Tile, kernel, ptx, reg, smem

    warp_size = 32
    block = _pick_block(dim)
    num_warps = block // warp_size
    items_per_thread = dim // block
    use_v4 = items_per_thread >= 4 and items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(rows, dim, f32), Tile(dim, f32), Tile(dim, f32)),
        out_specs=(Tile(rows, dim, f32),),
        grid=(rows, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def layer_norm(x, weight, bias, out):
        partials = smem.alloc(f32, (num_warps, 2))
        stats = smem.alloc(f32, (2, 1))
        px, pw, pb, po = ptx.global_ptrs(x, weight, bias, out)

        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & (warp_size - 1)
        warp_id = tid >> 5

        row_byte_off = row * (dim * 4)
        px += row_byte_off
        po += row_byte_off

        x_vals = reg.array(f32, items_per_thread)
        sum_x = reg.scalar(f32, init=0.0)
        sum_x2 = reg.scalar(f32, init=0.0)

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                ptr = px + idx * 4
                ptx.inst.ld.global_.v4.f32(
                    [x_vals[j * 4], x_vals[j * 4 + 1], x_vals[j * 4 + 2], x_vals[j * 4 + 3]],
                    ptx.addr(ptr),
                )
                for sub in range(4):
                    ptx.inst.add.f32(sum_x, sum_x, x_vals[j * 4 + sub])
                    ptx.inst.fma.rn.f32(
                        sum_x2,
                        x_vals[j * 4 + sub],
                        x_vals[j * 4 + sub],
                        sum_x2,
                    )
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                ptr = px + idx * 4
                ptx.inst.ld.global_.f32(x_vals[i], ptx.addr(ptr))
                ptx.inst.add.f32(sum_x, sum_x, x_vals[i])
                ptx.inst.fma.rn.f32(sum_x2, x_vals[i], x_vals[i], sum_x2)

        ptx.warp.reduce_sum(sum_x)
        ptx.warp.reduce_sum(sum_x2)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = sum_x
            partials[warp_id, 1] = sum_x2
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            block_sum_sq = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
                ptx.inst.add.f32(block_sum_sq, block_sum_sq, partials[i, 1])
            stats[0, 0] = block_sum
            stats[1, 0] = block_sum_sq
        ptx.bar.sync(0)

        ptx.inst.mov.f32(sum_x, stats[0, 0])
        ptx.inst.mov.f32(sum_x2, stats[1, 0])

        inv_dim = reg.scalar(f32, init=1.0 / dim)
        mean = reg.scalar(f32)
        ptx.inst.mul.f32(mean, sum_x, inv_dim)
        mean_sq = reg.scalar(f32)
        ptx.inst.mul.f32(mean_sq, mean, mean)
        ex2 = reg.scalar(f32)
        ptx.inst.mul.f32(ex2, sum_x2, inv_dim)
        var = reg.scalar(f32)
        ptx.inst.sub.f32(var, ex2, mean_sq)
        eps_reg = reg.scalar(f32, init=eps)
        ptx.inst.add.f32(var, var, eps_reg)
        rstd = reg.scalar(f32)
        ptx.inst.rsqrt.approx.f32(rstd, var)

        if use_v4:
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * 4
                w_vals = [reg.scalar(f32) for _ in range(4)]
                b_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(w_vals, ptx.addr(pw + off))
                ptx.inst.ld.global_.v4.f32(b_vals, ptx.addr(pb + off))
                y_vals = []
                for sub in range(4):
                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, x_vals[j * 4 + sub], mean)
                    y = reg.scalar(f32)
                    ptx.inst.mul.f32(y, diff, rstd)
                    ptx.inst.fma.rn.f32(y, y, w_vals[sub], b_vals[sub])
                    y_vals.append(y)
                ptx.inst.st.global_.v4.f32(ptx.addr(po + off), y_vals)
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                off = idx * 4
                w_val = reg.scalar(f32)
                b_val = reg.scalar(f32)
                ptx.inst.ld.global_.f32(w_val, ptx.addr(pw + off))
                ptx.inst.ld.global_.f32(b_val, ptx.addr(pb + off))
                diff = reg.scalar(f32)
                ptx.inst.sub.f32(diff, x_vals[i], mean)
                y = reg.scalar(f32)
                ptx.inst.mul.f32(y, diff, rstd)
                ptx.inst.fma.rn.f32(y, y, w_val, b_val)
                ptx.inst.st.global_.f32(ptx.addr(po + off), y)
        ptx.ret()

    return layer_norm


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


def attention_mask_from_lengths_or_torch(lengths: Tensor, max_length: int) -> Tensor:
    if max_length <= 0:
        return torch.empty((lengths.size(0), 0), device=lengths.device, dtype=torch.long)
    if not _is_cuda_int64(lengths) or lengths.dim() != 1:
        return (
            torch.arange(max_length, device=lengths.device).unsqueeze(0)
            < lengths.to(dtype=torch.long).unsqueeze(1)
        ).to(dtype=torch.long)
    try:
        kernel = _build_attention_mask_kernel(lengths.size(0), int(max_length), _arch_for(lengths))
        return kernel(lengths)
    except Exception:
        return (
            torch.arange(max_length, device=lengths.device).unsqueeze(0)
            < lengths.to(dtype=torch.long).unsqueeze(1)
        ).to(dtype=torch.long)


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


def layer_norm_or_torch(
    x: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    if (
        len(normalized_shape) != 1
        or weight is None
        or bias is None
        or not _is_inference_cuda_f32(x, weight, bias)
        or x.dim() < 1
    ):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    flat, original_shape = _flat_2d(x)
    if flat.size(1) != normalized_shape[0]:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    try:
        kernel = _build_layer_norm_kernel(
            flat.size(0),
            flat.size(1),
            float(eps),
            _arch_for(x),
        )
        return kernel(flat, weight, bias).reshape(original_shape)
    except Exception:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


def masked_mean_or_torch(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    if (
        _PYPTX_DISABLED
        or torch.is_grad_enabled()
        or hidden.dim() != 3
        or attention_mask.shape != hidden.shape[:2]
        or not (
            hidden.is_cuda
            and hidden.dtype == torch.float32
            and hidden.is_contiguous()
            and _is_cuda_int64(attention_mask)
        )
    ):
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom
    try:
        kernel = _build_masked_mean_kernel(
            hidden.size(0),
            hidden.size(1),
            hidden.size(2),
            _arch_for(hidden),
        )
        return kernel(hidden, attention_mask)
    except Exception:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom
