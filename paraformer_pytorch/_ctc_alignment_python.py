from __future__ import annotations

import torch


def ctc_viterbi_alignment_python(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Best CTC alignment for one utterance."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [time, vocab_with_blank]")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [target_len]")

    time, _ = log_probs.shape
    if targets.numel() == 0:
        return torch.full((time,), blank_id, dtype=torch.long, device=log_probs.device)

    extended = torch.full(
        (targets.numel() * 2 + 1,),
        blank_id,
        dtype=torch.long,
        device=targets.device,
    )
    extended[1::2] = targets
    states = extended.numel()

    neg_inf = torch.finfo(log_probs.dtype).min
    dp = torch.full((time, states), neg_inf, dtype=log_probs.dtype, device=log_probs.device)
    back = torch.full((time, states), -1, dtype=torch.long, device=log_probs.device)

    dp[0, 0] = log_probs[0, blank_id]
    if states > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for t in range(1, time):
        for s in range(states):
            candidates = [(dp[t - 1, s], s)]
            if s - 1 >= 0:
                candidates.append((dp[t - 1, s - 1], s - 1))
            if (
                s - 2 >= 0
                and extended[s] != blank_id
                and extended[s] != extended[s - 2]
            ):
                candidates.append((dp[t - 1, s - 2], s - 2))

            scores = torch.stack([score for score, _ in candidates])
            best_idx = int(torch.argmax(scores).item())
            best_score, best_state = candidates[best_idx]
            dp[t, s] = best_score + log_probs[t, extended[s]]
            back[t, s] = best_state

    final_state = states - 1
    if states > 1 and dp[-1, states - 2] > dp[-1, final_state]:
        final_state = states - 2

    state_path = torch.empty((time,), dtype=torch.long, device=log_probs.device)
    state_path[-1] = final_state
    for t in range(time - 1, 0, -1):
        state_path[t - 1] = back[t, state_path[t]]

    return extended.to(log_probs.device)[state_path]


def batch_ctc_viterbi_alignments_python(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Pad-aligned batch wrapper around `ctc_viterbi_alignment_python`."""
    alignments = torch.full(
        (log_probs.size(0), log_probs.size(1)),
        blank_id,
        dtype=torch.long,
        device=log_probs.device,
    )
    for b in range(log_probs.size(0)):
        t_len = int(input_lengths[b].item())
        y_len = int(target_lengths[b].item())
        alignments[b, :t_len] = ctc_viterbi_alignment_python(
            log_probs[b, :t_len],
            targets[b, :y_len],
            blank_id,
        )
    return alignments


def batch_uniform_alignments(
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    max_input_len: int,
    blank_id: int,
) -> torch.Tensor:
    """Cheap teacher-forced CTC-style alignments for diagnostic overfit runs."""
    alignments = torch.full(
        (targets.size(0), max_input_len),
        blank_id,
        dtype=torch.long,
        device=targets.device,
    )
    for b in range(targets.size(0)):
        t_len = int(input_lengths[b].item())
        y_len = int(target_lengths[b].item())
        if y_len == 0 or t_len == 0:
            continue
        extended = torch.full(
            (y_len * 2 + 1,),
            blank_id,
            dtype=torch.long,
            device=targets.device,
        )
        extended[1::2] = targets[b, :y_len]
        state_idx = torch.div(
            torch.arange(t_len, device=targets.device) * extended.numel(),
            t_len,
            rounding_mode="floor",
        ).clamp_max(extended.numel() - 1)
        alignments[b, :t_len] = extended[state_idx]
    return alignments
