from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

_PYPTX_DISABLE_VALUE = os.environ.get("SQUEEZEFORMER_DISABLE_PYPTX", "1").lower()
_PYPTX_DISABLED = _PYPTX_DISABLE_VALUE not in {"0", "false", "no", "off"}
_BLOCK_CANDIDATES = (1024, 512, 256, 128, 64, 32)
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_PYPTX_IMPORT_READY = False
_LOGGER = logging.getLogger(__name__)
_LOGGED_KERNEL_USES: set[tuple[object, ...]] = set()


def _log_kernel_use_once(name: str, *shape: object) -> None:
    key = (name, *shape)
    if key in _LOGGED_KERNEL_USES:
        return
    _LOGGED_KERNEL_USES.add(key)
    _LOGGER.info("using pyptx %s kernel shape=%s", name, shape)


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
        tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous()
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


def _is_cuda_bool(x: Tensor) -> bool:
    if _PYPTX_DISABLED:
        return False
    if not (x.is_cuda and x.dtype == torch.bool and x.is_contiguous()):
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


def _flat_1d(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    return x.reshape(-1), original_shape


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
def _build_sequence_mask_kernel(batch: int, time: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import pred, s64, u8, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    items_per_thread = (time + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, s64),),
        out_specs=(Tile(batch, time, pred),),
        grid=(batch, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def sequence_mask(lengths, out):
        pl, po = ptx.global_ptrs(lengths, out)
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())

        length = reg.scalar(s64)
        ptx.inst.ld.global_.s64(length, ptx.addr(pl + row * 8))
        out_row = po + row * time
        one = reg.scalar(u8, init=1)
        zero = reg.scalar(u8, init=0)

        for i in range(items_per_thread):
            col = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, col, time)
            col_s64 = reg.scalar(s64)
            ptx.inst.cvt.s64.u32(col_s64, col)
            is_valid = reg.scalar(pred)
            ptx.inst.setp.lt.s64(is_valid, col_s64, length)
            value = reg.scalar(u8)
            ptx.selp(u8, value, one, zero, is_valid)
            ptx.inst.st.global_.u8(ptx.addr(out_row + col), value, pred=in_bounds)
        ptx.ret()

    return sequence_mask


@lru_cache(maxsize=128)
def _build_attention_bool_mask_kernel(batch: int, time: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import pred, s64, u8, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    grid_z = (time + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, s64),),
        out_specs=(Tile(batch, time, time, pred),),
        grid=(batch, time, grid_z),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def attention_bool_mask(lengths, out):
        pl, po = ptx.global_ptrs(lengths, out)
        batch_idx = reg.scalar(u32)
        ptx.inst.mov.u32(batch_idx, ptx.special.ctaid.x())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        q = reg.scalar(u32)
        ptx.inst.mov.u32(q, ptx.special.ctaid.y())
        k_tile = reg.scalar(u32)
        ptx.inst.mov.u32(k_tile, ptx.special.ctaid.z())

        length = reg.scalar(s64)
        ptx.inst.ld.global_.s64(length, ptx.addr(pl + batch_idx * 8))
        out_row = po + (batch_idx * time + q) * time
        one = reg.scalar(u8, init=1)
        zero = reg.scalar(u8, init=0)

        k = k_tile * block + tid
        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, k, time)
        q_s64 = reg.scalar(s64)
        k_s64 = reg.scalar(s64)
        ptx.inst.cvt.s64.u32(q_s64, q)
        ptx.inst.cvt.s64.u32(k_s64, k)
        q_valid = reg.scalar(pred)
        k_valid = reg.scalar(pred)
        ptx.inst.setp.lt.s64(q_valid, q_s64, length)
        ptx.inst.setp.lt.s64(k_valid, k_s64, length)
        q_word = reg.scalar(u32)
        k_word = reg.scalar(u32)
        valid_word = reg.scalar(u32)
        ptx.selp(u32, q_word, 1, 0, q_valid)
        ptx.selp(u32, k_word, 1, 0, k_valid)
        ptx.inst.and_.b32(valid_word, q_word, k_word)
        is_valid = reg.scalar(pred)
        ptx.inst.setp.ne.u32(is_valid, valid_word, 0)
        value = reg.scalar(u8)
        ptx.selp(u8, value, one, zero, is_valid)
        ptx.inst.st.global_.u8(ptx.addr(out_row + k), value, pred=in_bounds)
        ptx.ret()

    return attention_bool_mask


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
def _build_apply_time_mask_kernel(batch: int, time: int, dim: int, layout: str, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    feature_tiles = (dim + block - 1) // block
    time_tiles = (time + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, time, dim, f32), Tile(batch, time, pred)),
        out_specs=(Tile(batch, time, dim, f32),),
        grid=(batch, time, feature_tiles),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def apply_btd(x, mask, out):
        px, pm, po = ptx.global_ptrs(x, mask, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        sample = reg.scalar(u32)
        ptx.inst.mov.u32(sample, ptx.special.ctaid.x())
        time_idx = reg.scalar(u32)
        ptx.inst.mov.u32(time_idx, ptx.special.ctaid.y())
        feature_tile = reg.scalar(u32)
        ptx.inst.mov.u32(feature_tile, ptx.special.ctaid.z())
        zero = reg.scalar(f32, init=0.0)
        feature = feature_tile * block + tid
        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, feature, dim)
        bt = sample * time + time_idx
        linear = bt * dim + feature
        mask_value = reg.scalar(u32, init=0)
        ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + bt), pred=in_bounds)
        keep = reg.scalar(pred)
        ptx.inst.setp.ne.u32(keep, mask_value, 0)
        x_value = reg.scalar(f32, init=0.0)
        ptx.inst.ld.global_.f32(x_value, ptx.addr(px + linear * 4), pred=in_bounds)
        y_value = reg.scalar(f32)
        ptx.selp(f32, y_value, x_value, zero, keep)
        ptx.inst.st.global_.f32(ptx.addr(po + linear * 4), y_value, pred=in_bounds)
        ptx.ret()

    @kernel(
        in_specs=(Tile(batch, dim, time, f32), Tile(batch, time, pred)),
        out_specs=(Tile(batch, dim, time, f32),),
        grid=(batch, dim, time_tiles),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def apply_bdt(x, mask, out):
        px, pm, po = ptx.global_ptrs(x, mask, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        sample = reg.scalar(u32)
        ptx.inst.mov.u32(sample, ptx.special.ctaid.x())
        feature = reg.scalar(u32)
        ptx.inst.mov.u32(feature, ptx.special.ctaid.y())
        time_tile = reg.scalar(u32)
        ptx.inst.mov.u32(time_tile, ptx.special.ctaid.z())
        zero = reg.scalar(f32, init=0.0)
        t = time_tile * block + tid
        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, t, time)
        mask_idx = sample * time + t
        linear = (sample * dim + feature) * time + t
        mask_value = reg.scalar(u32, init=0)
        ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + mask_idx), pred=in_bounds)
        keep = reg.scalar(pred)
        ptx.inst.setp.ne.u32(keep, mask_value, 0)
        x_value = reg.scalar(f32, init=0.0)
        ptx.inst.ld.global_.f32(x_value, ptx.addr(px + linear * 4), pred=in_bounds)
        y_value = reg.scalar(f32)
        ptx.selp(f32, y_value, x_value, zero, keep)
        ptx.inst.st.global_.f32(ptx.addr(po + linear * 4), y_value, pred=in_bounds)
        ptx.ret()

    return apply_bdt if layout == "bdt" else apply_btd


@lru_cache(maxsize=128)
def _build_time_recovery_repeat_kernel(
    batch: int, source_time: int, target_time: int, dim: int, stride: int, arch: str
):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    feature_tiles = (dim + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, source_time, dim, f32),),
        out_specs=(Tile(batch, target_time, dim, f32),),
        grid=(batch, target_time, feature_tiles),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def repeat_recover(x, out):
        px, po = ptx.global_ptrs(x, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        sample = reg.scalar(u32)
        ptx.inst.mov.u32(sample, ptx.special.ctaid.x())
        target_t = reg.scalar(u32)
        ptx.inst.mov.u32(target_t, ptx.special.ctaid.y())
        feature_tile = reg.scalar(u32)
        ptx.inst.mov.u32(feature_tile, ptx.special.ctaid.z())
        max_source = source_time - 1
        feature = feature_tile * block + tid
        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, feature, dim)
        source_t = reg.scalar(u32)
        ptx.inst.div.u32(source_t, target_t, stride)
        over = reg.scalar(pred)
        ptx.inst.setp.gt.u32(over, source_t, max_source)
        clipped_source = reg.scalar(u32)
        ptx.selp(u32, clipped_source, max_source, source_t, over)
        src_idx = (sample * source_time + clipped_source) * dim + feature
        linear = (sample * target_time + target_t) * dim + feature
        value = reg.scalar(f32, init=0.0)
        ptx.inst.ld.global_.f32(value, ptx.addr(px + src_idx * 4), pred=in_bounds)
        ptx.inst.st.global_.f32(ptx.addr(po + linear * 4), value, pred=in_bounds)
        ptx.ret()

    return repeat_recover


@lru_cache(maxsize=128)
def _build_ctc_log_prob_frame_stats_kernel(
    batch: int,
    time: int,
    vocab: int,
    blank_id: int,
    arch: str,
):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, s64, u32

    from pyptx import Tile, kernel, ptx, reg, smem

    warp_size = 32
    block = 256
    num_warps = block // warp_size
    items_per_thread = (vocab + block - 1) // block
    min_f32 = -3.4028234663852886e38
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(batch, time, vocab, f32), Tile(batch, s64)),
        out_specs=(Tile(batch, time, 4, f32),),
        grid=(batch, time, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def ctc_log_prob_frame_stats(log_probs, lengths, out):
        partials = smem.alloc(f32, (num_warps, 2))
        stats = smem.alloc(f32, (2, 1))
        plp, plen, po = ptx.global_ptrs(log_probs, lengths, out)

        batch_idx = reg.scalar(u32)
        ptx.inst.mov.u32(batch_idx, ptx.special.ctaid.x())
        time_idx = reg.scalar(u32)
        ptx.inst.mov.u32(time_idx, ptx.special.ctaid.y())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & (warp_size - 1)
        warp_id = tid >> 5

        length = reg.scalar(s64)
        ptx.inst.ld.global_.s64(length, ptx.addr(plen + batch_idx * 8))
        time_s64 = reg.scalar(s64)
        ptx.inst.cvt.s64.u32(time_s64, time_idx)
        is_valid_frame = reg.scalar(pred)
        ptx.inst.setp.lt.s64(is_valid_frame, time_s64, length)

        frame_base = (batch_idx * time + time_idx) * vocab
        blank_log_prob = reg.scalar(f32, init=min_f32)
        ptx.inst.ld.global_.f32(
            blank_log_prob,
            ptx.addr(plp + (frame_base + blank_id) * 4),
            pred=is_valid_frame,
        )

        top_any = reg.scalar(f32, init=min_f32)
        top_nonblank = reg.scalar(f32, init=min_f32)
        for i in range(items_per_thread):
            token = tid if i == 0 else tid + (i * block)
            in_vocab = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_vocab, token, vocab)
            value = reg.scalar(f32, init=min_f32)
            ptx.inst.ld.global_.f32(value, ptx.addr(plp + (frame_base + token) * 4), pred=in_vocab)
            ptx.inst.max.f32(top_any, top_any, value)

            is_blank = reg.scalar(pred)
            ptx.inst.setp.eq.u32(is_blank, token, blank_id)
            is_nonblank_word = reg.scalar(u32)
            ptx.selp(u32, is_nonblank_word, 0, 1, is_blank)
            is_nonblank = reg.scalar(pred)
            ptx.inst.setp.ne.u32(is_nonblank, is_nonblank_word, 0)
            nonblank_value = reg.scalar(f32)
            ptx.selp(f32, nonblank_value, value, min_f32, is_nonblank)
            ptx.inst.max.f32(top_nonblank, top_nonblank, nonblank_value)

        ptx.warp.reduce_max(top_any)
        ptx.warp.reduce_max(top_nonblank)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = top_any
            partials[warp_id, 1] = top_nonblank
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_top_any = reg.scalar(f32, init=min_f32)
            block_top_nonblank = reg.scalar(f32, init=min_f32)
            for i in range(num_warps):
                ptx.inst.max.f32(block_top_any, block_top_any, partials[i, 0])
                ptx.inst.max.f32(block_top_nonblank, block_top_nonblank, partials[i, 1])
            stats[0, 0] = block_top_any
            stats[1, 0] = block_top_nonblank
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            ptx.inst.mov.f32(top_nonblank, stats[1, 0])
            blank_is_argmax = reg.scalar(pred)
            ptx.inst.setp.ge.f32(blank_is_argmax, blank_log_prob, top_nonblank)

            valid_value = reg.scalar(f32)
            ptx.selp(f32, valid_value, 1.0, 0.0, is_valid_frame)
            argmax_blank_value = reg.scalar(f32)
            ptx.selp(f32, argmax_blank_value, 1.0, 0.0, blank_is_argmax)
            ptx.inst.mul.f32(argmax_blank_value, argmax_blank_value, valid_value)

            blank_exp_arg = reg.scalar(f32)
            ptx.inst.mul.f32(blank_exp_arg, blank_log_prob, _LOG2E)
            blank_prob = reg.scalar(f32)
            ptx.inst.ex2.approx.f32(blank_prob, blank_exp_arg)
            ptx.inst.mul.f32(blank_prob, blank_prob, valid_value)

            nonblank_exp_arg = reg.scalar(f32)
            ptx.inst.mul.f32(nonblank_exp_arg, top_nonblank, _LOG2E)
            top_nonblank_prob = reg.scalar(f32)
            ptx.inst.ex2.approx.f32(top_nonblank_prob, nonblank_exp_arg)
            ptx.inst.mul.f32(top_nonblank_prob, top_nonblank_prob, valid_value)

            out_base = (batch_idx * time + time_idx) * 4
            ptx.inst.st.global_.f32(ptx.addr(po + (out_base + 0) * 4), blank_prob)
            ptx.inst.st.global_.f32(ptx.addr(po + (out_base + 1) * 4), valid_value)
            ptx.inst.st.global_.f32(ptx.addr(po + (out_base + 2) * 4), argmax_blank_value)
            ptx.inst.st.global_.f32(ptx.addr(po + (out_base + 3) * 4), top_nonblank_prob)
        ptx.ret()

    return ctc_log_prob_frame_stats


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


@lru_cache(maxsize=128)
def _build_swoosh_kernel(
    total: int, offset: float, linear_scale: float, constant: float, arch: str
):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    grid_x = (total + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(total, f32),),
        out_specs=(Tile(total, f32),),
        grid=(grid_x, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def swoosh(x, out):
        px, po = ptx.global_ptrs(x, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        block_idx = reg.scalar(u32)
        ptx.inst.mov.u32(block_idx, ptx.special.ctaid.x())
        idx = block_idx * block + tid
        one = reg.scalar(f32, init=1.0)
        log2e = reg.scalar(f32, init=_LOG2E)
        ln2 = reg.scalar(f32, init=_LN2)
        offset_reg = reg.scalar(f32, init=offset)
        linear_scale_reg = reg.scalar(f32, init=linear_scale)
        constant_reg = reg.scalar(f32, init=constant)

        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, idx, total)
        xv = reg.scalar(f32, init=0.0)
        ptx.inst.ld.global_.f32(xv, ptx.addr(px + idx * 4), pred=in_bounds)
        shifted = reg.scalar(f32)
        ptx.inst.sub.f32(shifted, xv, offset_reg)
        abs_shifted = reg.scalar(f32)
        ptx.inst.abs.f32(abs_shifted, shifted)
        neg_abs_shifted = reg.scalar(f32)
        ptx.inst.neg.f32(neg_abs_shifted, abs_shifted)
        exp_arg = reg.scalar(f32)
        ptx.inst.mul.f32(exp_arg, neg_abs_shifted, log2e)
        exp_value = reg.scalar(f32)
        ptx.inst.ex2.approx.f32(exp_value, exp_arg)
        one_plus = reg.scalar(f32)
        ptx.inst.add.f32(one_plus, one, exp_value)
        log2_value = reg.scalar(f32)
        ptx.inst.lg2.approx.f32(log2_value, one_plus)
        log1p_exp = reg.scalar(f32)
        ptx.inst.mul.f32(log1p_exp, log2_value, ln2)
        zero = reg.scalar(f32, init=0.0)
        max_shifted = reg.scalar(f32)
        ptx.inst.max.f32(max_shifted, shifted, zero)
        softplus = reg.scalar(f32)
        ptx.inst.add.f32(softplus, max_shifted, log1p_exp)
        y = reg.scalar(f32)
        ptx.inst.fma.rn.f32(y, xv, linear_scale_reg, softplus)
        ptx.inst.sub.f32(y, y, constant_reg)
        ptx.inst.st.global_.f32(ptx.addr(po + idx * 4), y, pred=in_bounds)
        ptx.ret()

    return swoosh


@lru_cache(maxsize=128)
def _build_gated_linear_unit_kernel(m: int, f2: int, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, u32

    from pyptx import Tile, kernel, ptx, reg

    block = 256
    f = f2 // 2
    total = m * f
    grid_x = (total + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(m, f2, f32),),
        out_specs=(Tile(m, f, f32),),
        grid=(grid_x, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def gated_linear_unit(projected, out):
        pp, po = ptx.global_ptrs(projected, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        block_idx = reg.scalar(u32)
        ptx.inst.mov.u32(block_idx, ptx.special.ctaid.x())
        linear = block_idx * block + tid
        neg_log2e = reg.scalar(f32, init=-_LOG2E)
        one = reg.scalar(f32, init=1.0)

        in_bounds = reg.scalar(pred)
        ptx.inst.setp.lt.u32(in_bounds, linear, total)
        row = reg.scalar(u32)
        ptx.inst.div.u32(row, linear, f)
        col = linear - row * f
        row_base = row * f2
        x_value = reg.scalar(f32, init=0.0)
        gate_value = reg.scalar(f32, init=0.0)
        ptx.inst.ld.global_.f32(
            x_value,
            ptx.addr(pp + (row_base + col) * 4),
            pred=in_bounds,
        )
        ptx.inst.ld.global_.f32(
            gate_value,
            ptx.addr(pp + (row_base + f + col) * 4),
            pred=in_bounds,
        )
        neg_gate = reg.scalar(f32)
        ptx.inst.mul.f32(neg_gate, gate_value, neg_log2e)
        exp_neg = reg.scalar(f32)
        ptx.inst.ex2.approx.f32(exp_neg, neg_gate)
        denom = reg.scalar(f32)
        ptx.inst.add.f32(denom, one, exp_neg)
        sigm = reg.scalar(f32)
        ptx.inst.rcp.approx.f32(sigm, denom)
        y = reg.scalar(f32)
        ptx.inst.mul.f32(y, x_value, sigm)
        ptx.inst.st.global_.f32(ptx.addr(po + linear * 4), y, pred=in_bounds)
        ptx.ret()

    return gated_linear_unit


@lru_cache(maxsize=128)
def _build_bias_norm_kernel(rows: int, dim: int, eps: float, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx.types import f32, pred, u32

    from pyptx import Tile, kernel, ptx, reg, smem

    warp_size = 32
    block = 256
    num_warps = block // warp_size
    items_per_thread = (dim + block - 1) // block
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(Tile(rows, dim, f32), Tile(dim, f32), Tile(1, f32)),
        out_specs=(Tile(rows, dim, f32),),
        grid=(rows, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def bias_norm(x, bias, log_scale, out):
        partials = smem.alloc(f32, (num_warps, 1))
        stats = smem.alloc(f32, (1, 1))
        px, pb, pls, po = ptx.global_ptrs(x, bias, log_scale, out)
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & (warp_size - 1)
        warp_id = tid >> 5
        row_byte_off = row * (dim * 4)
        px += row_byte_off
        po += row_byte_off

        sum_sq = reg.scalar(f32, init=0.0)
        for i in range(items_per_thread):
            idx = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, idx, dim)
            xv = reg.scalar(f32, init=0.0)
            bv = reg.scalar(f32, init=0.0)
            ptx.inst.ld.global_.f32(xv, ptx.addr(px + idx * 4), pred=in_bounds)
            ptx.inst.ld.global_.f32(bv, ptx.addr(pb + idx * 4), pred=in_bounds)
            diff = reg.scalar(f32)
            ptx.inst.sub.f32(diff, xv, bv)
            ptx.inst.fma.rn.f32(sum_sq, diff, diff, sum_sq)

        ptx.warp.reduce_sum(sum_sq)
        with ptx.if_(lane == 0):
            partials[warp_id, 0] = sum_sq
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
            stats[0, 0] = block_sum
        ptx.bar.sync(0)

        ptx.inst.mov.f32(sum_sq, stats[0, 0])
        inv_dim = reg.scalar(f32, init=1.0 / dim)
        mean_sq = reg.scalar(f32)
        ptx.inst.mul.f32(mean_sq, sum_sq, inv_dim)
        eps_reg = reg.scalar(f32, init=eps)
        ptx.inst.add.f32(mean_sq, mean_sq, eps_reg)
        rstd = reg.scalar(f32)
        ptx.inst.rsqrt.approx.f32(rstd, mean_sq)
        ls = reg.scalar(f32)
        ptx.inst.ld.global_.f32(ls, ptx.addr(pls))
        exp_arg = reg.scalar(f32)
        ptx.inst.mul.f32(exp_arg, ls, _LOG2E)
        scale = reg.scalar(f32)
        ptx.inst.ex2.approx.f32(scale, exp_arg)
        ptx.inst.mul.f32(scale, scale, rstd)

        for i in range(items_per_thread):
            idx = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, idx, dim)
            xv = reg.scalar(f32, init=0.0)
            ptx.inst.ld.global_.f32(xv, ptx.addr(px + idx * 4), pred=in_bounds)
            y = reg.scalar(f32)
            ptx.inst.mul.f32(y, xv, scale)
            ptx.inst.st.global_.f32(ptx.addr(po + idx * 4), y, pred=in_bounds)
        ptx.ret()

    return bias_norm


def _disable_kernel_builders_for_torch_compile() -> None:
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:
        return
    for name in (
        "_build_attention_mask_kernel",
        "_build_sequence_mask_kernel",
        "_build_attention_bool_mask_kernel",
        "_build_scale_bias_kernel",
        "_build_masked_mean_kernel",
        "_build_layer_norm_kernel",
        "_build_apply_time_mask_kernel",
        "_build_time_recovery_repeat_kernel",
        "_build_ctc_log_prob_frame_stats_kernel",
        "_build_silu_kernel",
        "_build_swoosh_kernel",
        "_build_gated_linear_unit_kernel",
        "_build_bias_norm_kernel",
    ):
        globals()[name] = disable(
            globals()[name],
            recursive=True,
            reason="pyptx kernel builders use lru_cache and PTX codegen setup outside Dynamo graphs",
        )


_disable_kernel_builders_for_torch_compile()


def _launch_pyptx_kernel(kernel, *args):
    return kernel(*args)


_torch_compile_disable = getattr(getattr(torch, "compiler", None), "disable", None)
if _torch_compile_disable is not None:
    _launch_pyptx_kernel = _torch_compile_disable(
        _launch_pyptx_kernel,
        recursive=True,
        reason="pyptx Kernel.__call__ uses launch machinery outside Dynamo graphs",
    )


def _arch_for(x: Tensor) -> str:
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major >= 10:
        return "sm_100a"
    return "sm_90a"


def attention_mask_from_lengths_or_torch(lengths: Tensor, max_length: int) -> Tensor:
    if max_length <= 0:
        return torch.empty((lengths.size(0), 0), device=lengths.device, dtype=torch.long)
    if not _is_cuda_int64(lengths) or lengths.dim() != 1:
        _LOGGER.debug(
            "using torch attention_mask fallback shape=(%s, %s) device=%s dtype=%s",
            lengths.size(0),
            max_length,
            lengths.device,
            lengths.dtype,
        )
        return (
            torch.arange(max_length, device=lengths.device).unsqueeze(0)
            < lengths.to(dtype=torch.long).unsqueeze(1)
        ).to(dtype=torch.long)
    try:
        kernel = _build_attention_mask_kernel(lengths.size(0), int(max_length), _arch_for(lengths))
        _log_kernel_use_once("attention_mask", lengths.size(0), int(max_length), _arch_for(lengths))
        return _launch_pyptx_kernel(kernel, lengths)
    except Exception as exc:
        _LOGGER.debug(
            "using torch attention_mask fallback after pyptx failure shape=(%s, %s): %s",
            lengths.size(0),
            max_length,
            exc,
        )
        return (
            torch.arange(max_length, device=lengths.device).unsqueeze(0)
            < lengths.to(dtype=torch.long).unsqueeze(1)
        ).to(dtype=torch.long)


def sequence_mask_or_torch(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    if max_length <= 0:
        return torch.empty((lengths.size(0), 0), device=lengths.device, dtype=torch.bool)
    lengths = lengths.to(dtype=torch.long).contiguous()
    if not _is_cuda_int64(lengths) or lengths.dim() != 1:
        return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    try:
        kernel = _build_sequence_mask_kernel(lengths.size(0), int(max_length), _arch_for(lengths))
        _log_kernel_use_once("sequence_mask", lengths.size(0), int(max_length), _arch_for(lengths))
        return _launch_pyptx_kernel(kernel, lengths)
    except Exception as exc:
        _LOGGER.debug(
            "using torch sequence_mask fallback after pyptx failure shape=(%s, %s): %s",
            lengths.size(0),
            max_length,
            exc,
        )
        return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def squeezeformer_attention_mask_or_torch(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    if max_length <= 0:
        return torch.empty(
            (lengths.size(0), 0, 0),
            device=lengths.device,
            dtype=torch.bool,
        )
    lengths = lengths.to(dtype=torch.long).contiguous()
    if not _is_cuda_int64(lengths) or lengths.dim() != 1:
        sequence_mask = sequence_mask_or_torch(lengths, max_length=max_length)
        return sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
    try:
        kernel = _build_attention_bool_mask_kernel(
            lengths.size(0),
            int(max_length),
            _arch_for(lengths),
        )
        _log_kernel_use_once(
            "squeezeformer_attention_mask",
            lengths.size(0),
            int(max_length),
            _arch_for(lengths),
        )
        return _launch_pyptx_kernel(kernel, lengths)
    except Exception as exc:
        _LOGGER.debug(
            "using torch squeezeformer_attention_mask fallback after pyptx failure shape=(%s, %s): %s",
            lengths.size(0),
            max_length,
            exc,
        )
        sequence_mask = sequence_mask_or_torch(lengths, max_length=max_length)
        return sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)


def scale_bias_or_torch(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    if not _is_inference_cuda_f32(x, scale, bias) or x.dim() < 1:
        _LOGGER.debug(
            "using torch scale_bias fallback shape=%s device=%s dtype=%s",
            tuple(x.shape),
            x.device,
            x.dtype,
        )
        return x * scale + bias
    flat, original_shape = _flat_2d(x)
    try:
        kernel = _build_scale_bias_kernel(flat.size(0), flat.size(1), _arch_for(x))
        _log_kernel_use_once("scale_bias", flat.size(0), flat.size(1), _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat, scale, bias).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch scale_bias fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return x * scale + bias


def silu_or_torch(x: Tensor) -> Tensor:
    if not _is_inference_cuda_f32(x) or x.dim() < 1:
        _LOGGER.debug(
            "using torch silu fallback shape=%s device=%s dtype=%s",
            tuple(x.shape),
            x.device,
            x.dtype,
        )
        return F.silu(x)
    flat, original_shape = _flat_2d(x)
    try:
        kernel = _build_silu_kernel(flat.size(0), flat.size(1), _arch_for(x))
        _log_kernel_use_once("silu", flat.size(0), flat.size(1), _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch silu fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return F.silu(x)


def swoosh_l_or_torch(x: Tensor) -> Tensor:
    return _swoosh_or_torch(x, offset=4.0, linear_scale=-0.08, constant=0.035, name="swoosh_l")


def swoosh_r_or_torch(x: Tensor) -> Tensor:
    return _swoosh_or_torch(
        x,
        offset=1.0,
        linear_scale=-0.08,
        constant=0.313261687,
        name="swoosh_r",
    )


def _swoosh_or_torch(
    x: Tensor,
    *,
    offset: float,
    linear_scale: float,
    constant: float,
    name: str,
) -> Tensor:
    def torch_swoosh() -> Tensor:
        return F.softplus(x - offset) + linear_scale * x - constant

    if not _is_inference_cuda_f32(x) or x.dim() < 1:
        return torch_swoosh()
    flat, original_shape = _flat_1d(x)
    try:
        kernel = _build_swoosh_kernel(
            flat.numel(),
            float(offset),
            float(linear_scale),
            float(constant),
            _arch_for(x),
        )
        _log_kernel_use_once(name, flat.numel(), _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch %s fallback after pyptx failure shape=%s: %s",
            name,
            original_shape,
            exc,
        )
        return torch_swoosh()


def gated_linear_unit_or_torch(projected: Tensor) -> Tensor:
    if projected.size(-1) % 2 != 0:
        x, gate = projected.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
    if not _is_inference_cuda_f32(projected) or projected.dim() < 1:
        x, gate = projected.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
    flat, original_shape = _flat_2d(projected)
    try:
        kernel = _build_gated_linear_unit_kernel(flat.size(0), flat.size(1), _arch_for(projected))
        _log_kernel_use_once(
            "gated_linear_unit",
            flat.size(0),
            flat.size(1),
            _arch_for(projected),
        )
        return _launch_pyptx_kernel(kernel, flat).reshape(*original_shape[:-1], original_shape[-1] // 2)
    except Exception as exc:
        _LOGGER.debug(
            "using torch gated_linear_unit fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        x, gate = projected.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


def bias_norm_or_torch(x: Tensor, bias: Tensor, log_scale: Tensor, eps: float) -> Tensor:
    def torch_bias_norm() -> Tensor:
        view_shape = (*([1] * (x.ndim - 1)), x.size(-1))
        bias_view = bias.view(view_shape)
        rms = (x - bias_view).pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return x / rms * log_scale.exp()

    if (
        x.dim() < 1
        or x.size(-1) != bias.numel()
        or log_scale.numel() != 1
        or not _is_inference_cuda_f32(x, bias, log_scale)
    ):
        return torch_bias_norm()
    flat, original_shape = _flat_2d(x)
    try:
        kernel = _build_bias_norm_kernel(flat.size(0), flat.size(1), float(eps), _arch_for(x))
        _log_kernel_use_once("bias_norm", flat.size(0), flat.size(1), float(eps), _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat, bias, log_scale.reshape(1)).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch bias_norm fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return torch_bias_norm()


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
        _LOGGER.debug(
            "using torch layer_norm fallback shape=%s normalized_shape=%s device=%s dtype=%s",
            tuple(x.shape),
            normalized_shape,
            x.device,
            x.dtype,
        )
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    flat, original_shape = _flat_2d(x)
    if flat.size(1) != normalized_shape[0]:
        _LOGGER.debug(
            "using torch layer_norm fallback for mismatched shape=%s normalized_shape=%s",
            original_shape,
            normalized_shape,
        )
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    try:
        kernel = _build_layer_norm_kernel(
            flat.size(0),
            flat.size(1),
            float(eps),
            _arch_for(x),
        )
        _log_kernel_use_once("layer_norm", flat.size(0), flat.size(1), float(eps), _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat, weight, bias).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch layer_norm fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


def apply_time_mask_or_torch(x: Tensor, mask: Tensor, *, layout: str = "btd") -> Tensor:
    if (
        layout not in {"btd", "bdt"}
        or torch.is_grad_enabled()
        or x.dim() != 3
        or not _is_inference_cuda_f32(x)
        or not _is_cuda_bool(mask)
    ):
        if layout == "bdt":
            return x * mask.unsqueeze(1).to(dtype=x.dtype)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)
    batch = x.size(0)
    time = x.size(2) if layout == "bdt" else x.size(1)
    dim = x.size(1) if layout == "bdt" else x.size(2)
    if mask.shape != (batch, time):
        if layout == "bdt":
            return x * mask.unsqueeze(1).to(dtype=x.dtype)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)
    try:
        kernel = _build_apply_time_mask_kernel(batch, time, dim, layout, _arch_for(x))
        _log_kernel_use_once("apply_time_mask", batch, time, dim, layout, _arch_for(x))
        return _launch_pyptx_kernel(kernel, x, mask)
    except Exception as exc:
        _LOGGER.debug(
            "using torch apply_time_mask fallback after pyptx failure shape=%s layout=%s: %s",
            tuple(x.shape),
            layout,
            exc,
        )
        if layout == "bdt":
            return x * mask.unsqueeze(1).to(dtype=x.dtype)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


def time_recovery_repeat_or_torch(x: Tensor, target_length: int, stride: int) -> Tensor:
    if (
        torch.is_grad_enabled()
        or x.dim() != 3
        or not _is_inference_cuda_f32(x)
        or target_length <= 0
    ):
        repeated = torch.repeat_interleave(x, repeats=stride, dim=1)
        if repeated.size(1) < target_length:
            pad_length = target_length - repeated.size(1)
            return F.pad(repeated, (0, 0, 0, pad_length), mode="replicate")
        return repeated[:, :target_length, :]
    try:
        kernel = _build_time_recovery_repeat_kernel(
            x.size(0),
            x.size(1),
            int(target_length),
            x.size(2),
            int(stride),
            _arch_for(x),
        )
        _log_kernel_use_once(
            "time_recovery_repeat",
            x.size(0),
            x.size(1),
            int(target_length),
            x.size(2),
            int(stride),
            _arch_for(x),
        )
        return _launch_pyptx_kernel(kernel, x)
    except Exception as exc:
        _LOGGER.debug(
            "using torch time_recovery_repeat fallback after pyptx failure shape=%s target=%s: %s",
            tuple(x.shape),
            target_length,
            exc,
        )
        repeated = torch.repeat_interleave(x, repeats=stride, dim=1)
        if repeated.size(1) < target_length:
            pad_length = target_length - repeated.size(1)
            return F.pad(repeated, (0, 0, 0, pad_length), mode="replicate")
        return repeated[:, :target_length, :]


def ctc_log_prob_frame_stats_or_torch(
    log_probs: Tensor,
    output_lengths: Tensor,
    *,
    blank_id: int,
) -> Tensor:
    def torch_frame_stats() -> Tensor:
        valid_mask = torch.arange(log_probs.size(1), device=output_lengths.device).unsqueeze(
            0
        ) < output_lengths.to(dtype=torch.long).unsqueeze(1)
        blank_probabilities = log_probs[..., blank_id].exp()
        argmax_blank = log_probs.argmax(dim=-1).eq(blank_id).to(dtype=torch.float32)
        nonblank_log_probs = log_probs.clone()
        nonblank_log_probs[..., blank_id] = float("-inf")
        top_nonblank = nonblank_log_probs.max(dim=-1).values.exp()
        valid = valid_mask.to(dtype=torch.float32)
        return torch.stack(
            (
                blank_probabilities * valid,
                valid,
                argmax_blank * valid,
                top_nonblank * valid,
            ),
            dim=-1,
        )

    if (
        torch.is_grad_enabled()
        or log_probs.dim() != 3
        or not _is_inference_cuda_f32(log_probs)
        or not _is_cuda_int64(output_lengths.contiguous())
        or output_lengths.dim() != 1
        or output_lengths.size(0) != log_probs.size(0)
    ):
        return torch_frame_stats()
    try:
        lengths = output_lengths.to(dtype=torch.long).contiguous()
        kernel = _build_ctc_log_prob_frame_stats_kernel(
            log_probs.size(0),
            log_probs.size(1),
            log_probs.size(2),
            int(blank_id),
            _arch_for(log_probs),
        )
        _log_kernel_use_once(
            "ctc_log_prob_frame_stats",
            log_probs.size(0),
            log_probs.size(1),
            log_probs.size(2),
            int(blank_id),
            _arch_for(log_probs),
        )
        return _launch_pyptx_kernel(kernel, log_probs, lengths)
    except Exception as exc:
        _LOGGER.debug(
            "using torch ctc_log_prob_frame_stats fallback after pyptx failure shape=%s: %s",
            tuple(log_probs.shape),
            exc,
        )
        return torch_frame_stats()


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
        _LOGGER.debug(
            "using torch masked_mean fallback hidden_shape=%s mask_shape=%s device=%s dtype=%s",
            tuple(hidden.shape),
            tuple(attention_mask.shape),
            hidden.device,
            hidden.dtype,
        )
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
        _log_kernel_use_once(
            "masked_mean",
            hidden.size(0),
            hidden.size(1),
            hidden.size(2),
            _arch_for(hidden),
        )
        return _launch_pyptx_kernel(kernel, hidden, attention_mask)
    except Exception as exc:
        _LOGGER.debug(
            "using torch masked_mean fallback after pyptx failure hidden_shape=%s: %s",
            tuple(hidden.shape),
            exc,
        )
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom
