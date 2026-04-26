from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from squeezeformer_pytorch.pyptx_kernels import layer_norm_silu_scale_or_torch

from .ctc_alignment import batch_ctc_viterbi_alignments, batch_uniform_alignments
from .model import ConformerBlock, ConvSubsampling, ParaformerV2Config, lengths_to_padding_mask

try:
    dynamo_disable = torch._dynamo.disable
except AttributeError:

    def dynamo_disable(fn):  # type: ignore[no-redef]
        return fn


@dataclass(frozen=True)
class BetterParaformerV2Config(ParaformerV2Config):
    shallow_ctc_loss_weight: float = 0.3
    boundary_loss_weight: float = 0.1
    refinement_loss_weight: float = 0.3
    confidence_threshold: float = 0.55
    low_confidence_threshold: float = 0.7

    @classmethod
    def from_variant(
        cls, variant: str, **overrides: int | float | None
    ) -> "BetterParaformerV2Config":
        base = ParaformerV2Config.from_variant(variant)
        config = {
            "architecture": "paraformer_better",
            "input_dim": base.input_dim,
            "vocab_size": base.vocab_size,
            "encoder_dim": base.encoder_dim,
            "decoder_dim": base.decoder_dim,
            "encoder_layers": base.encoder_layers,
            "decoder_layers": base.decoder_layers,
            "encoder_ff_dim": base.encoder_ff_dim,
            "decoder_ff_dim": base.decoder_ff_dim,
            "attention_heads": base.attention_heads,
            "conv_kernel_size": base.conv_kernel_size,
            "dropout": base.dropout,
            "blank_id": base.blank_id,
        }
        config.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**config)


class MultiResolutionConformerEncoder(nn.Module):
    def __init__(self, config: BetterParaformerV2Config) -> None:
        super().__init__()
        self.subsampling = ConvSubsampling(config.input_dim, config.encoder_dim)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    config.encoder_dim,
                    config.encoder_ff_dim,
                    config.attention_heads,
                    config.conv_kernel_size,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.shallow_index = max(0, (config.encoder_layers // 2) - 1)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, lengths = self.subsampling(features, lengths)
        mask = lengths_to_padding_mask(lengths, x.size(1))
        shallow = None
        for index, layer in enumerate(self.layers):
            x = layer(x, mask)
            if index == self.shallow_index:
                shallow = x
        if shallow is None:
            shallow = x
        return shallow, x, lengths


class BetterParaformerV2(nn.Module):
    """Alternative Paraformer-v2 variant implementing the README follow-up ideas."""

    def __init__(self, config: BetterParaformerV2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = MultiResolutionConformerEncoder(config)
        self.shallow_ctc_projection = nn.Linear(config.encoder_dim, config.ctc_vocab_size)
        self.final_ctc_projection = nn.Linear(config.encoder_dim, config.ctc_vocab_size)
        self.boundary_head = nn.Sequential(
            nn.Linear(config.encoder_dim * 2, config.encoder_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_dim, 1),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(config.ctc_vocab_size * 2 + 4, config.decoder_dim),
            nn.LayerNorm(config.decoder_dim),
            nn.SiLU(),
        )
        self.memory_projection = (
            nn.Identity()
            if config.encoder_dim == config.decoder_dim
            else nn.Linear(config.encoder_dim, config.decoder_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.decoder_projection = nn.Linear(config.decoder_dim, config.vocab_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        refinement_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.refinement_decoder = nn.TransformerDecoder(refinement_layer, num_layers=1)
        self.refinement_projection = nn.Linear(config.decoder_dim, config.vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        alignment_mode: str = "viterbi",
        alignment_backend: str = "auto",
    ) -> dict[str, torch.Tensor]:
        shallow_encoder_out, final_encoder_out, encoder_lengths = self.encoder(
            features, feature_lengths
        )
        shallow_logits = self.shallow_ctc_projection(shallow_encoder_out)
        final_logits = self.final_ctc_projection(final_encoder_out)
        shallow_log_probs = F.log_softmax(shallow_logits, dim=-1)
        final_log_probs = F.log_softmax(final_logits, dim=-1)

        if targets is not None and target_lengths is not None:
            if alignment_mode == "viterbi":
                alignments = batch_ctc_viterbi_alignments(
                    final_log_probs.detach(),
                    encoder_lengths,
                    targets,
                    target_lengths,
                    self.config.resolved_blank_id,
                    backend=alignment_backend,
                )
            elif alignment_mode == "uniform":
                alignments = batch_uniform_alignments(
                    encoder_lengths,
                    targets,
                    target_lengths,
                    final_log_probs.size(1),
                    self.config.resolved_blank_id,
                )
            else:
                raise ValueError(f"unknown alignment_mode: {alignment_mode}")
        else:
            alignments = final_logits.argmax(dim=-1)

        boundary_logits = self.boundary_head(
            torch.cat([shallow_encoder_out, final_encoder_out], dim=-1)
        ).squeeze(-1)
        query_features, query_lengths, query_confidences = compress_confidence_gated_queries(
            shallow_logits.softmax(dim=-1),
            final_logits.softmax(dim=-1),
            alignments,
            encoder_lengths,
            self.config.resolved_blank_id,
            boundary_logits.sigmoid(),
            self.config.confidence_threshold,
        )
        query_projection = self.query_projection
        projected_queries = query_projection[0](query_features)
        decoder_in = layer_norm_silu_scale_or_torch(
            projected_queries,
            query_projection[1].weight,
            query_projection[1].bias,
            query_confidences,
            query_projection[1].eps,
        )
        memory = self.memory_projection(final_encoder_out)
        tgt_mask = lengths_to_padding_mask(query_lengths, decoder_in.size(1))
        memory_mask = lengths_to_padding_mask(encoder_lengths, memory.size(1))
        decoder_out = self.decoder(
            decoder_in,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )
        decoder_logits = self.decoder_projection(decoder_out)
        token_confidences, token_ids = decoder_logits.softmax(dim=-1).max(dim=-1)
        low_confidence = token_confidences < self.config.low_confidence_threshold
        correction_seed = decoder_out + self.token_embedding(token_ids)
        refinement_out = self.refinement_decoder(
            correction_seed,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )
        refined_states = torch.where(low_confidence.unsqueeze(-1), refinement_out, decoder_out)
        refined_logits = self.refinement_projection(refined_states)

        return {
            "decoder_logits": refined_logits,
            "initial_decoder_logits": decoder_logits,
            "ctc_log_probs": final_log_probs,
            "shallow_ctc_log_probs": shallow_log_probs,
            "encoder_lengths": encoder_lengths,
            "query_lengths": query_lengths,
            "query_confidences": query_confidences,
            "alignments": alignments,
            "boundary_logits": boundary_logits,
        }

    def loss(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        alignment_mode: str = "viterbi",
        alignment_backend: str = "auto",
    ) -> dict[str, torch.Tensor]:
        out = self(
            features,
            feature_lengths,
            targets,
            target_lengths,
            alignment_mode,
            alignment_backend,
        )
        final_ctc_loss = F.ctc_loss(
            out["ctc_log_probs"].transpose(0, 1),
            targets,
            out["encoder_lengths"],
            target_lengths,
            blank=self.config.resolved_blank_id,
            zero_infinity=True,
        )
        shallow_ctc_loss = F.ctc_loss(
            out["shallow_ctc_log_probs"].transpose(0, 1),
            targets,
            out["encoder_lengths"],
            target_lengths,
            blank=self.config.resolved_blank_id,
            zero_infinity=True,
        )
        base_ce_loss = masked_cross_entropy(out["initial_decoder_logits"], targets, target_lengths)
        refined_ce_loss = masked_cross_entropy(out["decoder_logits"], targets, target_lengths)
        boundary_targets = build_boundary_targets(out["alignments"], out["encoder_lengths"])
        boundary_mask = ~lengths_to_padding_mask(
            out["encoder_lengths"], out["boundary_logits"].size(1)
        )
        boundary_loss = F.binary_cross_entropy_with_logits(
            out["boundary_logits"][boundary_mask],
            boundary_targets[boundary_mask],
        )

        total = (
            final_ctc_loss
            + self.config.shallow_ctc_loss_weight * shallow_ctc_loss
            + base_ce_loss
            + self.config.refinement_loss_weight * refined_ce_loss
            + self.config.boundary_loss_weight * boundary_loss
        )
        return {
            "loss": total,
            "ctc_loss": final_ctc_loss.detach(),
            "shallow_ctc_loss": shallow_ctc_loss.detach(),
            "ce_loss": refined_ce_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
        }


@dynamo_disable
def compress_confidence_gated_queries(
    shallow_posteriors: torch.Tensor,
    final_posteriors: torch.Tensor,
    alignments: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int,
    boundary_probs: torch.Tensor,
    confidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pieces: list[torch.Tensor] = []
    piece_confidences: list[torch.Tensor] = []
    piece_lengths: list[int] = []
    feature_dim = shallow_posteriors.size(-1) * 2 + 4

    for batch_index in range(shallow_posteriors.size(0)):
        time_length = int(lengths[batch_index].item())
        labels = alignments[batch_index, :time_length]
        shallow_frames = shallow_posteriors[batch_index, :time_length]
        final_frames = final_posteriors[batch_index, :time_length]
        boundaries = boundary_probs[batch_index, :time_length]
        raw_segments = nonblank_segments(labels, blank_id)
        if not raw_segments:
            pieces.append(shallow_frames.new_zeros(1, feature_dim))
            piece_confidences.append(shallow_frames.new_ones(1))
            piece_lengths.append(1)
            continue

        segment_features = []
        segment_confidences = []
        for segment_index, (start, end, label) in enumerate(raw_segments):
            shallow_label_confidence = shallow_frames[start:end, label]
            final_label_confidence = final_frames[start:end, label]
            frame_confidence = 0.5 * (shallow_label_confidence + final_label_confidence)
            gate = torch.sigmoid((frame_confidence - confidence_threshold) * 12.0)
            gate = gate + 0.05
            gate_sum = gate.sum().clamp_min(1e-6)
            shallow_pool = (shallow_frames[start:end] * gate.unsqueeze(-1)).sum(dim=0) / gate_sum
            final_pool = (final_frames[start:end] * gate.unsqueeze(-1)).sum(dim=0) / gate_sum
            segment_confidence = frame_confidence.mean()
            left_boundary = (
                boundaries[start - 1] if segment_index > 0 else boundaries.new_tensor(1.0)
            )
            right_boundary = boundaries[end - 1]
            smooth_left = (1.0 - left_boundary) * 0.25
            smooth_right = (1.0 - right_boundary) * 0.25
            feature = torch.cat(
                [
                    shallow_pool,
                    final_pool,
                    torch.stack(
                        [
                            segment_confidence,
                            shallow_label_confidence.mean(),
                            left_boundary,
                            right_boundary,
                        ]
                    ),
                ]
            )
            feature = feature * (1.0 + smooth_left + smooth_right)
            segment_features.append(feature)
            segment_confidences.append(segment_confidence.clamp(0.05, 1.0))

        stacked = torch.stack(segment_features)
        stacked_confidences = torch.stack(segment_confidences)
        pieces.append(stacked)
        piece_confidences.append(stacked_confidences)
        piece_lengths.append(stacked.size(0))

    max_length = max(piece_lengths)
    padded = shallow_posteriors.new_zeros(shallow_posteriors.size(0), max_length, feature_dim)
    padded_confidences = shallow_posteriors.new_zeros(shallow_posteriors.size(0), max_length)
    for batch_index, piece in enumerate(pieces):
        padded[batch_index, : piece.size(0)] = piece
        padded_confidences[batch_index, : piece.size(0)] = piece_confidences[batch_index]
    return (
        padded,
        torch.tensor(piece_lengths, dtype=torch.long, device=shallow_posteriors.device),
        padded_confidences,
    )


@dynamo_disable
def nonblank_segments(labels: torch.Tensor, blank_id: int) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = 0
    while start < labels.size(0):
        label = int(labels[start].item())
        end = start + 1
        while end < labels.size(0) and int(labels[end].item()) == label:
            end += 1
        if label != blank_id:
            segments.append((start, end, label))
        start = end
    return segments


def build_boundary_targets(alignments: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros_like(alignments, dtype=torch.float32)
    if alignments.size(1) > 1:
        changes = alignments[:, 1:] != alignments[:, :-1]
        targets[:, :-1] = changes.float()
    mask = lengths_to_padding_mask(lengths, alignments.size(1))
    targets[mask] = 0.0
    return targets


@dynamo_disable
def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    max_target = int(target_lengths.max().item())
    trimmed_logits = logits[:, :max_target]
    if trimmed_logits.size(1) < max_target:
        pad = logits.new_zeros(logits.size(0), max_target - trimmed_logits.size(1), logits.size(-1))
        trimmed_logits = torch.cat([trimmed_logits, pad], dim=1)
    target_mask = lengths_to_padding_mask(target_lengths, max_target)
    return F.cross_entropy(
        trimmed_logits[~target_mask],
        targets[:, :max_target][~target_mask],
    )
