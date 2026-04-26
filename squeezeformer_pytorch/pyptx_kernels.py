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
_PYPTX_TORCH_RUNTIME_READY = False
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


def _ensure_pyptx_torch_runtime_ready() -> None:
    global _PYPTX_TORCH_RUNTIME_READY
    if _PYPTX_TORCH_RUNTIME_READY or _PYPTX_DISABLED:
        return
    try:
        from pyptx import torch_support

        shim_path = torch_support._find_shim_path()
        if shim_path is None:
            try:
                from pyptx._shim.auto_build import try_auto_build

                shim_path = try_auto_build()
            except Exception as exc:
                _LOGGER.debug("pyptx shim auto-build failed: %s", exc)

        ext = torch_support._try_load_cpp_ext()
        if ext is not None and shim_path is not None:
            ext.load_shim(shim_path)
        _PYPTX_TORCH_RUNTIME_READY = True
    except Exception as exc:
        _LOGGER.debug("pyptx torch runtime setup failed: %s", exc)


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


def _is_inference_cuda_float(x: Tensor, *others: Tensor) -> bool:
    if _PYPTX_DISABLED or torch.is_grad_enabled():
        return False
    tensors = (x, *others)
    dtype = x.dtype
    if dtype not in {torch.float32, torch.bfloat16}:
        return False
    if not all(
        tensor.is_cuda and tensor.dtype == dtype and tensor.is_contiguous() for tensor in tensors
    ):
        return False
    major, _minor = torch.cuda.get_device_capability(x.device)
    return major >= 9


def _float_dtype_key(x: Tensor) -> str:
    if x.dtype == torch.bfloat16:
        return "bf16"
    if x.dtype == torch.float32:
        return "f32"
    raise ValueError(f"unsupported pyptx float dtype: {x.dtype}")


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


def _pick_scale_bias_block(n: int) -> int:
    try:
        return _pick_block(n)
    except ValueError:
        return 128 if n <= 256 else 256


def _pick_v4_block(n: int) -> int:
    block = _pick_v4_block_or_none(n)
    if block is not None:
        return block
    raise ValueError(f"dimension {n} is not supported by pyptx v4 kernels")


def _pick_v4_block_or_none(n: int) -> int | None:
    if n % 4 != 0:
        return None
    for block in (256, 128, 64, 32):
        if n % (block * 4) == 0:
            return block
    return None


def _flat_2d(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    return x.reshape(-1, original_shape[-1]), original_shape


def _flat_1d(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    return x.reshape(-1), original_shape


@lru_cache(maxsize=128)
def _build_scale_bias_kernel(m: int, f: int, dtype: str, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    block = _pick_scale_bias_block(f)
    items_per_thread = (f + block - 1) // block
    use_v4 = dtype == "f32" and f % (block * 4) == 0
    v4_iters = f // (block * 4) if use_v4 else 0
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred_guard=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred_guard)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred_guard)
        return value

    def store_data_f32(ptr, value, *, pred_guard=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred_guard)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred_guard)

    @kernel(
        in_specs=(Tile(m, f, data_t), Tile(f, data_t), Tile(f, data_t)),
        out_specs=(Tile(m, f, data_t),),
        grid=(m, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def scale_bias(x, scale, bias, out):
        px, ps, pb, po = ptx.global_ptrs(x, scale, bias, out)
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_byte_off = row * (f * elem_bytes)
        px += row_byte_off
        po += row_byte_off

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * elem_bytes
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
                in_bounds = reg.scalar(pred)
                ptx.inst.setp.lt.u32(in_bounds, idx, f)
                off = idx * elem_bytes
                xv = load_data_f32(px + off, pred_guard=in_bounds)
                sv = load_data_f32(ps + off, pred_guard=in_bounds)
                bv = load_data_f32(pb + off, pred_guard=in_bounds)
                y = reg.scalar(f32)
                ptx.inst.fma.rn.f32(y, xv, sv, bv)
                store_data_f32(po + off, y, pred_guard=in_bounds)
        ptx.ret()

    return scale_bias


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
    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.types import f32, pred, s64, u32

    warp_size = 32
    block = 256
    num_warps = block // warp_size
    items_per_thread = (vocab + block - 1) // block
    min_f32 = -3.4028234663852886e38
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

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
def _build_silu_time_mask_kernel(
    batch: int,
    time: int,
    dim: int,
    layout: str,
    dtype: str,
    arch: str,
):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    can_use_v4 = dtype == "f32"
    total = batch * time * dim
    scalar_block = 256
    scalar_grid_x = (total + scalar_block - 1) // scalar_block
    btd_v4_block = _pick_v4_block_or_none(dim) if can_use_v4 else None
    btd_use_v4 = btd_v4_block is not None
    btd_block = btd_v4_block if btd_v4_block is not None else scalar_block
    btd_v4_iters = dim // (btd_block * 4) if btd_use_v4 else 0
    bdt_v4_block = _pick_v4_block_or_none(time) if can_use_v4 else None
    bdt_use_v4 = bdt_v4_block is not None
    bdt_block = bdt_v4_block if bdt_v4_block is not None else scalar_block
    bdt_v4_iters = time // (bdt_block * 4) if bdt_use_v4 else 0
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred)
        return value

    def store_data_f32(ptr, value, *, pred=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred)

    def emit_silu(xv):
        neg_log2e = reg.scalar(f32, init=-_LOG2E)
        one = reg.scalar(f32, init=1.0)
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

    @kernel(
        in_specs=(Tile(batch, time, dim, data_t), Tile(batch, time, pred)),
        out_specs=(Tile(batch, time, dim, data_t),),
        grid=(batch * time if btd_use_v4 else scalar_grid_x, 1, 1),
        block=(btd_block, 1, 1),
        arch=arch,
        version=version,
    )
    def silu_mask_btd(x, mask, out):
        px, pm, po = ptx.global_ptrs(x, mask, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        block_idx = reg.scalar(u32)
        ptx.inst.mov.u32(block_idx, ptx.special.ctaid.x())
        zero = reg.scalar(f32, init=0.0)

        if btd_use_v4:
            row = block_idx
            row_byte_off = row * (dim * elem_bytes)
            px += row_byte_off
            po += row_byte_off
            mask_value = reg.scalar(u32, init=0)
            ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + row))
            keep = reg.scalar(pred)
            ptx.inst.setp.ne.u32(keep, mask_value, 0)
            elem_base = tid << 2
            for j in range(btd_v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * btd_block * 4)
                off = idx * elem_bytes
                x_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(x_vals, ptx.addr(px + off))
                y_vals = []
                for sub in range(4):
                    y_value = emit_silu(x_vals[sub])
                    ptx.selp(f32, y_value, y_value, zero, keep)
                    y_vals.append(y_value)
                ptx.inst.st.global_.v4.f32(ptx.addr(po + off), y_vals)
        else:
            linear = block_idx * scalar_block + tid
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, linear, total)
            bt = reg.scalar(u32)
            ptx.inst.div.u32(bt, linear, dim)
            mask_value = reg.scalar(u32, init=0)
            ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + bt), pred=in_bounds)
            keep = reg.scalar(pred)
            ptx.inst.setp.ne.u32(keep, mask_value, 0)
            x_value = reg.scalar(f32, init=0.0)
            x_value = load_data_f32(px + linear * elem_bytes, pred=in_bounds)
            y_value = emit_silu(x_value)
            ptx.selp(f32, y_value, y_value, zero, keep)
            store_data_f32(po + linear * elem_bytes, y_value, pred=in_bounds)
        ptx.ret()

    @kernel(
        in_specs=(Tile(batch, dim, time, data_t), Tile(batch, time, pred)),
        out_specs=(Tile(batch, dim, time, data_t),),
        grid=(batch * dim if bdt_use_v4 else scalar_grid_x, 1, 1),
        block=(bdt_block, 1, 1),
        arch=arch,
        version=version,
    )
    def silu_mask_bdt(x, mask, out):
        px, pm, po = ptx.global_ptrs(x, mask, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        block_idx = reg.scalar(u32)
        ptx.inst.mov.u32(block_idx, ptx.special.ctaid.x())
        zero = reg.scalar(f32, init=0.0)

        if bdt_use_v4:
            row = block_idx
            row_byte_off = row * (time * elem_bytes)
            px += row_byte_off
            po += row_byte_off
            sample = reg.scalar(u32)
            ptx.inst.div.u32(sample, row, dim)
            mask_row = sample * time
            elem_base = tid << 2
            for j in range(bdt_v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * bdt_block * 4)
                off = idx * elem_bytes
                x_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(x_vals, ptx.addr(px + off))
                y_vals = []
                for sub in range(4):
                    mask_value = reg.scalar(u32, init=0)
                    ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + mask_row + idx + sub))
                    keep = reg.scalar(pred)
                    ptx.inst.setp.ne.u32(keep, mask_value, 0)
                    y_value = emit_silu(x_vals[sub])
                    ptx.selp(f32, y_value, y_value, zero, keep)
                    y_vals.append(y_value)
                ptx.inst.st.global_.v4.f32(ptx.addr(po + off), y_vals)
        else:
            linear = block_idx * scalar_block + tid
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, linear, total)
            sample = reg.scalar(u32)
            ptx.inst.div.u32(sample, linear, dim * time)
            rem = linear - sample * (dim * time)
            t = reg.scalar(u32)
            ptx.inst.rem.u32(t, rem, time)
            mask_idx = sample * time + t
            mask_value = reg.scalar(u32, init=0)
            ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + mask_idx), pred=in_bounds)
            keep = reg.scalar(pred)
            ptx.inst.setp.ne.u32(keep, mask_value, 0)
            x_value = load_data_f32(px + linear * elem_bytes, pred=in_bounds)
            y_value = emit_silu(x_value)
            ptx.selp(f32, y_value, y_value, zero, keep)
            store_data_f32(po + linear * elem_bytes, y_value, pred=in_bounds)
        ptx.ret()

    return silu_mask_bdt if layout == "bdt" else silu_mask_btd


@lru_cache(maxsize=128)
def _build_conv_output_epilogue_kernel(batch: int, time: int, dim: int, dtype: str, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    can_use_v4 = dtype == "f32"
    v4_block = _pick_v4_block_or_none(dim) if can_use_v4 else None
    block = v4_block if v4_block is not None else 256
    use_v4 = v4_block is not None
    v4_iters = dim // (block * 4) if use_v4 else 0
    items_per_thread = (dim + block - 1) // block
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred)
        return value

    def store_data_f32(ptr, value, *, pred=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred)

    @kernel(
        in_specs=(
            Tile(batch, time, dim, data_t),
            Tile(batch, dim, time, data_t),
            Tile(batch, time, pred),
        ),
        out_specs=(Tile(batch, time, dim, data_t),),
        grid=(batch, time, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def conv_output_epilogue(residual, x, mask, out):
        pr, px, pm, po = ptx.global_ptrs(residual, x, mask, out)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        sample = reg.scalar(u32)
        ptx.inst.mov.u32(sample, ptx.special.ctaid.x())
        time_idx = reg.scalar(u32)
        ptx.inst.mov.u32(time_idx, ptx.special.ctaid.y())
        zero = reg.scalar(f32, init=0.0)

        bt = sample * time + time_idx
        residual_row = bt * dim
        x_sample_base = sample * (dim * time)
        mask_value = reg.scalar(u32, init=0)
        ptx.inst.ld.global_.u8(mask_value, ptx.addr(pm + bt))
        keep = reg.scalar(pred)
        ptx.inst.setp.ne.u32(keep, mask_value, 0)

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                feature = elem_base if j == 0 else elem_base + (j * block * 4)
                out_off = (residual_row + feature) * elem_bytes
                residual_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(residual_vals, ptx.addr(pr + out_off))
                y_vals = []
                for sub in range(4):
                    x_value = reg.scalar(f32, init=0.0)
                    x_idx = x_sample_base + (feature + sub) * time + time_idx
                    ptx.inst.ld.global_.f32(x_value, ptx.addr(px + x_idx * elem_bytes))
                    ptx.selp(f32, x_value, x_value, zero, keep)
                    y = reg.scalar(f32)
                    ptx.inst.add.f32(y, residual_vals[sub], x_value)
                    y_vals.append(y)
                ptx.inst.st.global_.v4.f32(ptx.addr(po + out_off), y_vals)
        else:
            for i in range(items_per_thread):
                feature = tid if i == 0 else tid + (i * block)
                in_bounds = reg.scalar(pred)
                ptx.inst.setp.lt.u32(in_bounds, feature, dim)
                residual_idx = residual_row + feature
                residual_value = reg.scalar(f32, init=0.0)
                x_value = reg.scalar(f32, init=0.0)
                residual_value = load_data_f32(pr + residual_idx * elem_bytes, pred=in_bounds)
                x_idx = x_sample_base + feature * time + time_idx
                x_value = load_data_f32(px + x_idx * elem_bytes, pred=in_bounds)
                ptx.selp(f32, x_value, x_value, zero, keep)
                y = reg.scalar(f32)
                ptx.inst.add.f32(y, residual_value, x_value)
                store_data_f32(po + residual_idx * elem_bytes, y, pred=in_bounds)
        ptx.ret()

    return conv_output_epilogue


@lru_cache(maxsize=128)
def _build_swoosh_kernel(
    total: int, offset: float, linear_scale: float, constant: float, dtype: str, arch: str
):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    block = 256
    grid_x = (total + block - 1) // block
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred)
        return value

    def store_data_f32(ptr, value, *, pred=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred)

    @kernel(
        in_specs=(Tile(total, data_t),),
        out_specs=(Tile(total, data_t),),
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
        xv = load_data_f32(px + idx * elem_bytes, pred=in_bounds)
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
        store_data_f32(po + idx * elem_bytes, y, pred=in_bounds)
        ptx.ret()

    return swoosh


@lru_cache(maxsize=128)
def _build_gated_linear_unit_kernel(m: int, f2: int, dtype: str, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    f = f2 // 2
    total = m * f
    scalar_block = 256
    scalar_grid_x = (total + scalar_block - 1) // scalar_block
    v4_block = _pick_v4_block_or_none(f) if dtype == "f32" else None
    use_v4 = v4_block is not None
    block = v4_block if v4_block is not None else scalar_block
    v4_iters = f // (block * 4) if use_v4 else 0
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred)
        return value

    def store_data_f32(ptr, value, *, pred=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred)

    @kernel(
        in_specs=(Tile(m, f2, data_t),),
        out_specs=(Tile(m, f, data_t),),
        grid=(m if use_v4 else scalar_grid_x, 1, 1),
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
        neg_log2e = reg.scalar(f32, init=-_LOG2E)
        one = reg.scalar(f32, init=1.0)

        def emit_one(x_value, gate_value):
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
            return y

        if use_v4:
            row = block_idx
            pp += row * (f2 * elem_bytes)
            po += row * (f * elem_bytes)
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * elem_bytes
                x_vals = [reg.scalar(f32) for _ in range(4)]
                gate_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(x_vals, ptx.addr(pp + off))
                ptx.inst.ld.global_.v4.f32(gate_vals, ptx.addr(pp + (f * elem_bytes) + off))
                ptx.inst.st.global_.v4.f32(
                    ptx.addr(po + off),
                    [emit_one(x_vals[sub], gate_vals[sub]) for sub in range(4)],
                )
        else:
            linear = block_idx * scalar_block + tid
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, linear, total)
            row = reg.scalar(u32)
            ptx.inst.div.u32(row, linear, f)
            col = linear - row * f
            row_base = row * f2
            x_value = load_data_f32(pp + (row_base + col) * elem_bytes, pred=in_bounds)
            gate_value = load_data_f32(pp + (row_base + f + col) * elem_bytes, pred=in_bounds)
            y = emit_one(x_value, gate_value)
            store_data_f32(po + linear * elem_bytes, y, pred=in_bounds)
        ptx.ret()

    return gated_linear_unit


@lru_cache(maxsize=128)
def _build_bias_norm_kernel(rows: int, dim: int, eps: float, dtype: str, arch: str):
    if not _ensure_pyptx_importable():
        raise RuntimeError("pyptx is not importable")
    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.types import b16, bf16, f32, pred, u32

    data_t = bf16 if dtype == "bf16" else f32
    elem_bytes = 2 if dtype == "bf16" else 4
    warp_size = 32
    block = 256
    num_warps = block // warp_size
    items_per_thread = (dim + block - 1) // block
    version = (8, 7) if arch.startswith(("sm_100", "sm_120")) else None

    def load_data_f32(ptr, *, pred=None):
        value = reg.scalar(f32, init=0.0)
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.ld.global_.b16(raw, ptx.addr(ptr), pred=pred)
            ptx.inst.cvt.f32.bf16(value, raw)
        else:
            ptx.inst.ld.global_.f32(value, ptx.addr(ptr), pred=pred)
        return value

    def store_data_f32(ptr, value, *, pred=None):
        if dtype == "bf16":
            raw = reg.scalar(b16, init=0)
            ptx.inst.cvt.rn.bf16.f32(raw, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr), raw, pred=pred)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr), value, pred=pred)

    @kernel(
        in_specs=(Tile(rows, dim, data_t), Tile(dim, data_t), Tile(1, data_t)),
        out_specs=(Tile(rows, dim, data_t),),
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
        row_byte_off = row * (dim * elem_bytes)
        px += row_byte_off
        po += row_byte_off

        sum_sq = reg.scalar(f32, init=0.0)
        for i in range(items_per_thread):
            idx = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, idx, dim)
            xv = load_data_f32(px + idx * elem_bytes, pred=in_bounds)
            bv = load_data_f32(pb + idx * elem_bytes, pred=in_bounds)
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
        ls = load_data_f32(pls)
        exp_arg = reg.scalar(f32)
        ptx.inst.mul.f32(exp_arg, ls, _LOG2E)
        scale = reg.scalar(f32)
        ptx.inst.ex2.approx.f32(scale, exp_arg)
        ptx.inst.mul.f32(scale, scale, rstd)

        for i in range(items_per_thread):
            idx = tid if i == 0 else tid + (i * block)
            in_bounds = reg.scalar(pred)
            ptx.inst.setp.lt.u32(in_bounds, idx, dim)
            xv = load_data_f32(px + idx * elem_bytes, pred=in_bounds)
            y = reg.scalar(f32)
            ptx.inst.mul.f32(y, xv, scale)
            store_data_f32(po + idx * elem_bytes, y, pred=in_bounds)
        ptx.ret()

    return bias_norm


def _disable_kernel_builders_for_torch_compile() -> None:
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:
        return
    for name in (
        "_build_scale_bias_kernel",
        "_build_ctc_log_prob_frame_stats_kernel",
        "_build_silu_time_mask_kernel",
        "_build_conv_output_epilogue_kernel",
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
    _ensure_pyptx_torch_runtime_ready()
    return kernel(*args)


_torch_compile_disable = getattr(getattr(torch, "compiler", None), "disable", None)
if _torch_compile_disable is not None:
    _launch_pyptx_kernel = _torch_compile_disable(
        _launch_pyptx_kernel,
        recursive=True,
        reason="pyptx Kernel.__call__ uses launch machinery outside Dynamo graphs",
    )


def _arch_for(x: Tensor) -> str:
    major, minor = torch.cuda.get_device_capability(x.device)
    if major >= 12:
        return f"sm_{major}{minor}"
    if major >= 10:
        return "sm_100a"
    return "sm_90a"


def scale_bias_or_torch(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    if not _is_inference_cuda_float(x, scale, bias) or x.dim() < 1:
        _LOGGER.debug(
            "using torch scale_bias fallback shape=%s device=%s dtype=%s",
            tuple(x.shape),
            x.device,
            x.dtype,
        )
        return x * scale + bias
    flat, original_shape = _flat_2d(x)
    try:
        dtype = _float_dtype_key(x)
        kernel = _build_scale_bias_kernel(flat.size(0), flat.size(1), dtype, _arch_for(x))
        _log_kernel_use_once("scale_bias", flat.size(0), flat.size(1), dtype, _arch_for(x))
        return _launch_pyptx_kernel(kernel, flat, scale, bias).reshape(original_shape)
    except Exception as exc:
        _LOGGER.debug(
            "using torch scale_bias fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return x * scale + bias


def silu_time_mask_or_torch(x: Tensor, mask: Tensor, *, layout: str = "bdt") -> Tensor:
    def torch_silu_mask() -> Tensor:
        y = F.silu(x)
        if layout == "bdt":
            return y * mask.unsqueeze(1).to(dtype=x.dtype)
        return y * mask.unsqueeze(-1).to(dtype=x.dtype)

    if (
        layout not in {"btd", "bdt"}
        or torch.is_grad_enabled()
        or x.dim() != 3
        or not _is_inference_cuda_float(x)
        or not _is_cuda_bool(mask)
    ):
        return torch_silu_mask()
    batch = x.size(0)
    time = x.size(2) if layout == "bdt" else x.size(1)
    dim = x.size(1) if layout == "bdt" else x.size(2)
    if mask.shape != (batch, time):
        return torch_silu_mask()
    try:
        dtype = _float_dtype_key(x)
        kernel = _build_silu_time_mask_kernel(batch, time, dim, layout, dtype, _arch_for(x))
        _log_kernel_use_once("silu_time_mask", batch, time, dim, layout, dtype, _arch_for(x))
        return _launch_pyptx_kernel(kernel, x, mask)
    except Exception as exc:
        _LOGGER.debug(
            "using torch silu_time_mask fallback after pyptx failure shape=%s layout=%s: %s",
            tuple(x.shape),
            layout,
            exc,
        )
        return torch_silu_mask()


def conv_output_epilogue_or_torch(residual: Tensor, x_bdt: Tensor, mask: Tensor) -> Tensor:
    def torch_epilogue() -> Tensor:
        masked = x_bdt * mask.unsqueeze(1).to(dtype=x_bdt.dtype)
        return residual + masked.transpose(1, 2)

    if (
        residual.dim() != 3
        or x_bdt.dim() != 3
        or residual.size(0) != x_bdt.size(0)
        or residual.size(1) != x_bdt.size(2)
        or residual.size(2) != x_bdt.size(1)
        or mask.shape != (residual.size(0), residual.size(1))
        or torch.is_grad_enabled()
        or not _is_inference_cuda_float(residual, x_bdt)
        or not _is_cuda_bool(mask)
    ):
        return torch_epilogue()
    try:
        dtype = _float_dtype_key(residual)
        kernel = _build_conv_output_epilogue_kernel(
            residual.size(0),
            residual.size(1),
            residual.size(2),
            dtype,
            _arch_for(residual),
        )
        _log_kernel_use_once(
            "conv_output_epilogue",
            residual.size(0),
            residual.size(1),
            residual.size(2),
            dtype,
            _arch_for(residual),
        )
        return _launch_pyptx_kernel(kernel, residual, x_bdt, mask)
    except Exception as exc:
        _LOGGER.debug(
            "using torch conv_output_epilogue fallback after pyptx failure residual_shape=%s "
            "x_shape=%s: %s",
            tuple(residual.shape),
            tuple(x_bdt.shape),
            exc,
        )
        return torch_epilogue()


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

    if not _is_inference_cuda_float(x) or x.dim() < 1:
        return torch_swoosh()
    flat, original_shape = _flat_1d(x)
    try:
        dtype = _float_dtype_key(x)
        kernel = _build_swoosh_kernel(
            flat.numel(),
            float(offset),
            float(linear_scale),
            float(constant),
            dtype,
            _arch_for(x),
        )
        _log_kernel_use_once(name, flat.numel(), dtype, _arch_for(x))
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
    if not _is_inference_cuda_float(projected) or projected.dim() < 1:
        x, gate = projected.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
    flat, original_shape = _flat_2d(projected)
    try:
        dtype = _float_dtype_key(projected)
        kernel = _build_gated_linear_unit_kernel(
            flat.size(0),
            flat.size(1),
            dtype,
            _arch_for(projected),
        )
        _log_kernel_use_once(
            "gated_linear_unit",
            flat.size(0),
            flat.size(1),
            dtype,
            _arch_for(projected),
        )
        return _launch_pyptx_kernel(kernel, flat).reshape(
            *original_shape[:-1], original_shape[-1] // 2
        )
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
        or not _is_inference_cuda_float(x, bias, log_scale)
    ):
        return torch_bias_norm()
    flat, original_shape = _flat_2d(x)
    try:
        dtype = _float_dtype_key(x)
        kernel = _build_bias_norm_kernel(
            flat.size(0),
            flat.size(1),
            float(eps),
            dtype,
            _arch_for(x),
        )
        _log_kernel_use_once(
            "bias_norm",
            flat.size(0),
            flat.size(1),
            float(eps),
            dtype,
            _arch_for(x),
        )
        return _launch_pyptx_kernel(kernel, flat, bias, log_scale.reshape(1)).reshape(
            original_shape
        )
    except Exception as exc:
        _LOGGER.debug(
            "using torch bias_norm fallback after pyptx failure shape=%s: %s",
            original_shape,
            exc,
        )
        return torch_bias_norm()


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
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom
