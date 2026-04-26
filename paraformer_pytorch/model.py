from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ctc_alignment import batch_ctc_viterbi_alignments, batch_uniform_alignments

try:
    dynamo_disable = torch._dynamo.disable
except AttributeError:

    def dynamo_disable(fn):  # type: ignore[no-redef]
        return fn


@dataclass(frozen=True)
class ParaformerV2Config:
    architecture: str = "paraformer"
    input_dim: int = 80
    vocab_size: int = 256
    encoder_dim: int = 256
    decoder_dim: int = 256
    encoder_layers: int = 12
    decoder_layers: int = 6
    encoder_ff_dim: int = 2048
    decoder_ff_dim: int = 2048
    attention_heads: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1
    blank_id: int | None = None

    @property
    def ctc_vocab_size(self) -> int:
        if self.blank_id is not None and 0 <= self.blank_id < self.vocab_size:
            return self.vocab_size
        return self.vocab_size + 1

    @property
    def resolved_blank_id(self) -> int:
        return self.vocab_size if self.blank_id is None else self.blank_id

    @classmethod
    def from_variant(cls, variant: str, **overrides: int | float | None) -> "ParaformerV2Config":
        presets = {
            "small": {
                "encoder_dim": 256,
                "decoder_dim": 256,
                "encoder_layers": 12,
                "decoder_layers": 6,
                "encoder_ff_dim": 2048,
                "decoder_ff_dim": 2048,
                "attention_heads": 4,
            },
            "medium": {
                "encoder_dim": 384,
                "decoder_dim": 384,
                "encoder_layers": 12,
                "decoder_layers": 6,
                "encoder_ff_dim": 2048,
                "decoder_ff_dim": 2048,
                "attention_heads": 6,
            },
            "large": {
                "encoder_dim": 512,
                "decoder_dim": 512,
                "encoder_layers": 12,
                "decoder_layers": 6,
                "encoder_ff_dim": 2048,
                "decoder_ff_dim": 2048,
                "attention_heads": 8,
            },
        }
        try:
            preset = presets[variant]
        except KeyError as exc:
            choices = ", ".join(sorted(presets))
            raise ValueError(f"unknown variant: {variant}. Expected one of: {choices}") from exc

        config = dict(preset)
        config.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**config)


class ConvSubsampling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(output_dim * ((input_dim + 3) // 4), output_dim)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.unsqueeze(1)
        x = self.conv(x)
        batch, channels, time, freq = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, time, channels * freq)
        lengths = torch.div(lengths + 3, 4, rounding_mode="floor")
        return self.proj(x), lengths.clamp_max(time)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, heads: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.ff1 = FeedForward(dim, ff_dim, dropout)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
            ),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2 = FeedForward(dim, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        attn_in = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(attn_out)
        conv_in = self.conv_norm(x).transpose(1, 2)
        x = x + self.conv(conv_in).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, config: ParaformerV2Config) -> None:
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

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.subsampling(features, lengths)
        mask = lengths_to_padding_mask(lengths, x.size(1))
        for layer in self.layers:
            x = layer(x, mask)
        return x, lengths


class ParaformerV2(nn.Module):
    """Paraformer-v2 core: CTC-derived decoder queries plus a non-causal decoder."""

    def __init__(self, config: ParaformerV2Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConformerEncoder(config)
        self.ctc_projection = nn.Linear(config.encoder_dim, config.ctc_vocab_size)
        self.posterior_embed = nn.Linear(config.ctc_vocab_size, config.decoder_dim, bias=False)
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

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        alignment_mode: str = "viterbi",
        alignment_backend: str = "auto",
    ) -> dict[str, torch.Tensor]:
        encoder_out, encoder_lengths = self.encoder(features, feature_lengths)
        ctc_logits = self.ctc_projection(encoder_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        if targets is not None and target_lengths is not None:
            if alignment_mode == "viterbi":
                alignments = batch_ctc_viterbi_alignments(
                    ctc_log_probs.detach(),
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
                    ctc_log_probs.size(1),
                    self.config.resolved_blank_id,
                )
            else:
                raise ValueError(f"unknown alignment_mode: {alignment_mode}")
        else:
            alignments = ctc_logits.argmax(dim=-1)

        compressed, compressed_lengths = compress_posteriors(
            ctc_logits.softmax(dim=-1),
            alignments,
            encoder_lengths,
            self.config.resolved_blank_id,
        )
        decoder_in = self.posterior_embed(compressed)
        memory = self.memory_projection(encoder_out)
        tgt_mask = lengths_to_padding_mask(compressed_lengths, decoder_in.size(1))
        memory_mask = lengths_to_padding_mask(encoder_lengths, memory.size(1))
        decoder_out = self.decoder(
            decoder_in,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )
        decoder_logits = self.decoder_projection(decoder_out)

        return {
            "decoder_logits": decoder_logits,
            "ctc_log_probs": ctc_log_probs,
            "encoder_lengths": encoder_lengths,
            "query_lengths": compressed_lengths,
            "alignments": alignments,
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
        ctc_loss = F.ctc_loss(
            out["ctc_log_probs"].transpose(0, 1),
            targets,
            out["encoder_lengths"],
            target_lengths,
            blank=self.config.resolved_blank_id,
            zero_infinity=True,
        )
        ce_loss = masked_cross_entropy(out["decoder_logits"], targets, target_lengths)
        return {
            "loss": ctc_loss + ce_loss,
            "ctc_loss": ctc_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


@dynamo_disable
def compress_posteriors(
    posteriors: torch.Tensor,
    alignments: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average consecutive frames with the same non-blank CTC label."""
    pieces: list[torch.Tensor] = []
    piece_lengths = []
    for b in range(posteriors.size(0)):
        t_len = int(lengths[b].item())
        labels = alignments[b, :t_len]
        frames = posteriors[b, :t_len]
        utterance = []
        start = 0
        while start < t_len:
            label = int(labels[start].item())
            end = start + 1
            while end < t_len and int(labels[end].item()) == label:
                end += 1
            if label != blank_id:
                utterance.append(frames[start:end].mean(dim=0))
            start = end
        if not utterance:
            utterance.append(frames.new_zeros(frames.size(-1)))
        stacked = torch.stack(utterance)
        pieces.append(stacked)
        piece_lengths.append(stacked.size(0))

    max_len = max(piece_lengths)
    padded = posteriors.new_zeros(posteriors.size(0), max_len, posteriors.size(-1))
    for b, piece in enumerate(pieces):
        padded[b, : piece.size(0)] = piece
    return padded, torch.tensor(piece_lengths, dtype=torch.long, device=posteriors.device)


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


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)
