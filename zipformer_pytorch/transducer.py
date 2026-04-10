from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def targets_to_padded(targets: Tensor, target_lengths: Tensor) -> Tensor:
    batch_size = int(target_lengths.numel())
    max_target_length = int(target_lengths.max().item()) if batch_size > 0 else 0
    padded = targets.new_zeros((batch_size, max_target_length))
    offset = 0
    for batch_index, target_length in enumerate(target_lengths.tolist()):
        length = int(target_length)
        if length > 0:
            padded[batch_index, :length] = targets[offset : offset + length]
        offset += length
    return padded


def make_decoder_inputs(targets_padded: Tensor, *, blank_id: int) -> Tensor:
    batch_size = int(targets_padded.size(0))
    blank_column = targets_padded.new_full((batch_size, 1), blank_id)
    return torch.cat((blank_column, targets_padded), dim=1)


def linear_prune_ranges(
    output_lengths: Tensor,
    target_lengths: Tensor,
    *,
    prune_range: int,
) -> Tensor:
    batch_size = int(output_lengths.numel())
    max_time = int(output_lengths.max().item()) if batch_size > 0 else 0
    if batch_size == 0 or max_time == 0:
        return output_lengths.new_zeros((batch_size, max_time, 0))

    effective_range = max(1, min(prune_range, int(target_lengths.max().item()) + 1))
    time_indices = torch.arange(max_time, device=output_lengths.device).unsqueeze(0)
    centers = (
        ((time_indices.to(torch.float32) + 0.5)
        * (target_lengths + 1).unsqueeze(1).to(torch.float32))
        / output_lengths.clamp_min(1).unsqueeze(1).to(torch.float32)
    ).floor().to(torch.long)
    half_width = effective_range // 2
    starts = (centers - half_width).clamp_min(0)
    max_start = (target_lengths + 1 - effective_range).clamp_min(0).unsqueeze(1)
    starts = torch.minimum(starts, max_start)
    offsets = torch.arange(effective_range, device=output_lengths.device).view(1, 1, -1)
    return starts.unsqueeze(-1) + offsets


def batched_rnnt_loss(
    blank_log_probs: Tensor,
    label_log_probs: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    batch_size, max_time, max_u_plus_one = blank_log_probs.shape
    max_u = max_u_plus_one - 1
    neg_inf = float("-inf")
    row = blank_log_probs.new_full((batch_size, max_u_plus_one), neg_inf)
    row[:, 0] = 0.0
    u_indices = torch.arange(max_u_plus_one, device=blank_log_probs.device).unsqueeze(0)

    for time_index in range(max_time):
        active_time = input_lengths > time_index
        closed_states = [row[:, 0]]
        if max_u > 0:
            for label_index in range(max_u):
                active_emit = active_time & (target_lengths > label_index)
                emit_scores = closed_states[-1] + label_log_probs[:, time_index, label_index]
                closed_states.append(
                    torch.logaddexp(
                        row[:, label_index + 1],
                        torch.where(
                            active_emit,
                            emit_scores,
                            emit_scores.new_full((batch_size,), neg_inf),
                        ),
                    )
                )
            closed_row = torch.stack(closed_states, dim=1)
        else:
            closed_row = row
        valid_blank = active_time.unsqueeze(1) & u_indices.le(target_lengths.unsqueeze(1))
        row = torch.where(
            valid_blank,
            closed_row + blank_log_probs[:, time_index, :],
            closed_row.new_full(closed_row.shape, neg_inf),
        )

    batch_indices = torch.arange(batch_size, device=blank_log_probs.device)
    final_scores = row[batch_indices, target_lengths]
    return (-final_scores / target_lengths.clamp_min(1).to(dtype=final_scores.dtype)).mean()


class StatelessDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ) -> None:
        super().__init__()
        if context_size < 1:
            raise ValueError(f"context_size must be >= 1, got {context_size}.")
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_dim)
        self.blank_id = int(blank_id)
        self.context_size = int(context_size)
        self.vocab_size = int(vocab_size)
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=max(1, decoder_dim // 4),
                bias=False,
            )
        else:
            self.conv = nn.Identity()

    def forward(self, tokens: Tensor, *, need_pad: bool = True) -> Tensor:
        tokens = tokens.to(torch.int64)
        if torch.jit.is_tracing():
            embedded = self.embedding(tokens)
        else:
            embedded = self.embedding(tokens.clamp(min=0)) * (tokens >= 0).unsqueeze(-1)
        if self.context_size > 1:
            embedded = embedded.transpose(1, 2)
            if need_pad:
                embedded = F.pad(embedded, (self.context_size - 1, 0))
            else:
                if embedded.size(-1) != self.context_size:
                    raise ValueError(
                        "Decoder inputs without padding must have exactly context_size steps."
                    )
            embedded = self.conv(embedded).transpose(1, 2)
        return F.relu(embedded)


class TransducerJoiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joiner_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joiner_dim)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: Tensor,
        decoder_out: Tensor,
        *,
        project_input: bool = True,
    ) -> Tensor:
        if project_input:
            combined = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            combined = encoder_out + decoder_out
        return self.output_linear(torch.tanh(combined))


@dataclass
class Hypothesis:
    tokens: list[int]
    log_prob: float

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(self.tokens)


def _update_hypothesis_pool(pool: dict[tuple[int, ...], Hypothesis], hypothesis: Hypothesis) -> None:
    existing = pool.get(hypothesis.key)
    if existing is None:
        pool[hypothesis.key] = hypothesis
        return
    current = torch.tensor(existing.log_prob, dtype=torch.float32)
    incoming = torch.tensor(hypothesis.log_prob, dtype=torch.float32)
    existing.log_prob = float(torch.logaddexp(current, incoming).item())


def transducer_greedy_search_batch(
    model: nn.Module,
    encoder_out: Tensor,
    encoder_out_lens: Tensor,
) -> list[list[int]]:
    hyps: list[list[int]] = []
    for sample_index, output_length in enumerate(encoder_out_lens.tolist()):
        hyps.append(
            transducer_greedy_search(
                model,
                encoder_out[sample_index : sample_index + 1, : int(output_length)],
            )
        )
    return hyps


def transducer_greedy_search(model: nn.Module, encoder_out: Tensor) -> list[int]:
    if encoder_out.ndim != 3 or encoder_out.size(0) != 1:
        raise ValueError("Greedy transducer decoding expects encoder_out with shape [1, T, C].")
    blank_id = int(model.decoder.blank_id)
    context_size = int(model.decoder.context_size)
    device = encoder_out.device
    decoder_input = torch.full(
        (1, context_size),
        blank_id,
        device=device,
        dtype=torch.int64,
    )
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    encoder_proj = model.joiner.encoder_proj(encoder_out)
    hypothesis = [blank_id] * context_size
    for frame_index in range(encoder_proj.size(1)):
        logits = model.joiner(
            encoder_proj[:, frame_index : frame_index + 1].unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        token_id = int(logits.argmax(dim=-1).item())
        if token_id == blank_id:
            continue
        hypothesis.append(token_id)
        decoder_input = torch.tensor(
            [hypothesis[-context_size:]],
            device=device,
            dtype=torch.int64,
        )
        decoder_out = model.decoder(decoder_input, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
    return hypothesis[context_size:]


def transducer_modified_beam_search_batch(
    model: nn.Module,
    encoder_out: Tensor,
    encoder_out_lens: Tensor,
    *,
    beam_size: int,
) -> list[list[int]]:
    hyps: list[list[int]] = []
    for sample_index, output_length in enumerate(encoder_out_lens.tolist()):
        hyps.append(
            transducer_modified_beam_search(
                model,
                encoder_out[sample_index : sample_index + 1, : int(output_length)],
                beam_size=beam_size,
            )
        )
    return hyps


def transducer_modified_beam_search(
    model: nn.Module,
    encoder_out: Tensor,
    *,
    beam_size: int = 4,
) -> list[int]:
    if encoder_out.ndim != 3 or encoder_out.size(0) != 1:
        raise ValueError("Beam search expects encoder_out with shape [1, T, C].")
    beam_size = max(1, int(beam_size))
    blank_id = int(model.decoder.blank_id)
    context_size = int(model.decoder.context_size)
    device = encoder_out.device
    encoder_proj = model.joiner.encoder_proj(encoder_out)
    hypotheses = {
        tuple([blank_id] * context_size): Hypothesis(
            tokens=[blank_id] * context_size,
            log_prob=0.0,
        )
    }

    for frame_index in range(encoder_proj.size(1)):
        next_hypotheses: dict[tuple[int, ...], Hypothesis] = {}
        for hypothesis in hypotheses.values():
            decoder_input = torch.tensor(
                [hypothesis.tokens[-context_size:]],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)
            logits = model.joiner(
                encoder_proj[:, frame_index : frame_index + 1].unsqueeze(2),
                decoder_out.unsqueeze(1),
                project_input=False,
            ).squeeze(0).squeeze(0).squeeze(0)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            topk_values, topk_indices = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

            blank_hypothesis = Hypothesis(
                tokens=hypothesis.tokens[:],
                log_prob=hypothesis.log_prob + float(log_probs[blank_id].item()),
            )
            _update_hypothesis_pool(next_hypotheses, blank_hypothesis)

            for log_prob, token_index in zip(topk_values.tolist(), topk_indices.tolist(), strict=True):
                token_id = int(token_index)
                if token_id == blank_id:
                    continue
                candidate = Hypothesis(
                    tokens=hypothesis.tokens + [token_id],
                    log_prob=hypothesis.log_prob + float(log_prob),
                )
                _update_hypothesis_pool(next_hypotheses, candidate)

        hypotheses = dict(
            sorted(
                next_hypotheses.items(),
                key=lambda item: item[1].log_prob,
                reverse=True,
            )[:beam_size]
        )

    best_hypothesis = max(
        hypotheses.values(),
        key=lambda hyp: hyp.log_prob / max(1, len(hyp.tokens) - context_size),
    )
    return best_hypothesis.tokens[context_size:]
