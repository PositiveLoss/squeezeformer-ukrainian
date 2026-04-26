from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _state_labels(
    targets_base,
    states,
    target_len,
    blank_id,
):
    odd = (states & 1) == 1
    target_idx = states // 2
    token = tl.load(
        targets_base + target_idx,
        mask=odd & (target_idx < target_len),
        other=blank_id,
    )
    return tl.where(odd, token, blank_id)


@triton.jit
def _forward_kernel(
    log_probs_ptr,
    input_lengths_ptr,
    targets_ptr,
    target_lengths_ptr,
    back_ptr,
    final_states_ptr,
    work_prev_ptr,
    work_curr_ptr,
    blank_id,
    max_time,
    vocab_size,
    max_target_len,
    max_states,
    BLOCK_STATES: tl.constexpr,
    MAX_TIME: tl.constexpr,
):
    pid = tl.program_id(0)
    states = tl.arange(0, BLOCK_STATES)
    valid_block = states < max_states

    t_len = tl.load(input_lengths_ptr + pid)
    y_len = tl.load(target_lengths_ptr + pid)
    states_count = y_len * 2 + 1
    valid_states = states < states_count

    targets_base = targets_ptr + pid * max_target_len
    labels = _state_labels(targets_base, states, y_len, blank_id)

    log_probs_base = log_probs_ptr + pid * max_time * vocab_size
    prev = work_prev_ptr + pid * BLOCK_STATES
    curr = work_curr_ptr + pid * BLOCK_STATES

    tl.store(prev + states, -1.0e30, mask=valid_block)
    tl.store(curr + states, -1.0e30, mask=valid_block)

    lp_blank = tl.load(
        log_probs_base + blank_id,
        mask=t_len > 0,
        other=-1.0e30,
    )
    first_target = tl.load(
        targets_base,
        mask=y_len > 0,
        other=blank_id,
    )
    lp_first = tl.load(
        log_probs_base + first_target,
        mask=(t_len > 0) & (states_count > 1),
        other=-1.0e30,
    )

    init_scores = tl.full((BLOCK_STATES,), -1.0e30, tl.float32)
    init_scores = tl.where(states == 0, lp_blank, init_scores)
    init_scores = tl.where((states == 1) & (states_count > 1), lp_first, init_scores)
    init_scores = tl.where(valid_states & (t_len > 0), init_scores, -1.0e30)
    tl.store(prev + states, init_scores, mask=valid_block)

    for t in range(1, MAX_TIME):
        prev_scores = tl.load(prev + states, mask=valid_block, other=-1.0e30)
        prev_left = tl.load(
            prev + states - 1,
            mask=valid_states & (states > 0),
            other=-1.0e30,
        )

        states_m2 = states - 2
        labels_m2 = _state_labels(targets_base, states_m2, y_len, blank_id)
        skip_allowed = valid_states & (states > 1) & (labels != blank_id) & (labels != labels_m2)
        prev_skip = tl.load(prev + states - 2, mask=skip_allowed, other=-1.0e30)

        best_scores = prev_scores
        best_states = states.to(tl.int64)

        left_better = prev_left > best_scores
        best_scores = tl.where(left_better, prev_left, best_scores)
        best_states = tl.where(left_better, (states - 1).to(tl.int64), best_states)

        skip_better = prev_skip > best_scores
        best_scores = tl.where(skip_better, prev_skip, best_scores)
        best_states = tl.where(skip_better, (states - 2).to(tl.int64), best_states)

        lp = tl.load(
            log_probs_base + t * vocab_size + labels,
            mask=valid_states & (t < t_len),
            other=0.0,
        )
        curr_scores = tl.where(valid_states & (t < t_len), best_scores + lp, -1.0e30)
        tl.store(curr + states, curr_scores, mask=valid_block)
        tl.store(
            back_ptr + pid * max_time * max_states + t * max_states + states,
            best_states,
            mask=valid_states & (t < t_len),
        )
        prev, curr = curr, prev

    final_scores = tl.load(prev + states, mask=valid_block, other=-1.0e30)
    score_last = tl.max(tl.where(states == (states_count - 1), final_scores, -1.0e30), axis=0)
    score_penult = tl.max(
        tl.where((states_count > 1) & (states == (states_count - 2)), final_scores, -1.0e30),
        axis=0,
    )
    final_state = tl.where((states_count > 1) & (score_penult > score_last), states_count - 2, states_count - 1)
    tl.store(final_states_ptr + pid, final_state)


@triton.jit
def _backtrace_kernel(
    targets_ptr,
    input_lengths_ptr,
    target_lengths_ptr,
    back_ptr,
    final_states_ptr,
    alignments_ptr,
    blank_id,
    max_time,
    max_target_len,
    max_states,
    MAX_TIME: tl.constexpr,
):
    pid = tl.program_id(0)
    t_len = tl.load(input_lengths_ptr + pid)
    y_len = tl.load(target_lengths_ptr + pid)
    state = tl.load(final_states_ptr + pid)

    targets_base = targets_ptr + pid * max_target_len
    align_base = alignments_ptr + pid * max_time
    back_base = back_ptr + pid * max_time * max_states

    for t in range(MAX_TIME - 1, -1, -1):
        if t < t_len:
            odd = (state & 1) == 1
            target_idx = state // 2
            token = tl.load(
                targets_base + target_idx,
                mask=odd & (target_idx < y_len),
                other=blank_id,
            )
            label = tl.where(odd, token, blank_id)
            tl.store(align_base + t, label)
            if t > 0:
                state = tl.load(back_base + t * max_states + state)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def batch_ctc_viterbi_alignments_triton(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    if not log_probs.is_cuda:
        raise ValueError("Triton CTC alignment backend requires CUDA tensors.")
    if log_probs.dtype != torch.float32:
        log_probs = log_probs.to(dtype=torch.float32)

    log_probs = log_probs.contiguous()
    input_lengths = input_lengths.to(device=log_probs.device, dtype=torch.long).contiguous()
    targets = targets.to(device=log_probs.device, dtype=torch.long).contiguous()
    target_lengths = target_lengths.to(device=log_probs.device, dtype=torch.long).contiguous()

    batch, max_time, vocab_size = log_probs.shape
    max_target_len = targets.size(1)
    max_states = max_target_len * 2 + 1
    block_states = _next_power_of_two(max_states)
    if block_states > 1024:
        raise RuntimeError(f"Triton CTC alignment backend only supports up to 1024 states, got {block_states}.")

    alignments = torch.full((batch, max_time), blank_id, dtype=torch.long, device=log_probs.device)
    back = torch.full((batch, max_time, max_states), -1, dtype=torch.long, device=log_probs.device)
    final_states = torch.zeros((batch,), dtype=torch.long, device=log_probs.device)
    work_prev = torch.empty((batch, block_states), dtype=torch.float32, device=log_probs.device)
    work_curr = torch.empty((batch, block_states), dtype=torch.float32, device=log_probs.device)

    grid = (batch,)
    _forward_kernel[grid](
        log_probs,
        input_lengths,
        targets,
        target_lengths,
        back,
        final_states,
        work_prev,
        work_curr,
        int(blank_id),
        max_time,
        vocab_size,
        max_target_len,
        max_states,
        BLOCK_STATES=block_states,
        MAX_TIME=max_time,
    )
    _backtrace_kernel[grid](
        targets,
        input_lengths,
        target_lengths,
        back,
        final_states,
        alignments,
        int(blank_id),
        max_time,
        max_target_len,
        max_states,
        MAX_TIME=max_time,
    )
    return alignments
