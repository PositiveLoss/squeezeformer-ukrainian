from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from torch import Tensor, nn
from torch.nn import functional as F

from .better_model import (
    BetterParaformerV2,
    BetterParaformerV2Config,
    build_boundary_targets,
)
from .better_model import (
    masked_cross_entropy as better_masked_cross_entropy,
)
from .model import (
    ParaformerV2,
    ParaformerV2Config,
    lengths_to_padding_mask,
    masked_cross_entropy,
)

_VARIANT_ALIASES = {
    "xs": "small",
    "s": "small",
    "sm": "medium",
    "m": "medium",
    "ml": "large",
    "l": "large",
}


def _paraformer_variant_name(name: str) -> str:
    return _VARIANT_ALIASES.get(name, name)


def paraformer_variant(
    name: str,
    *,
    input_dim: int,
    vocab_size: int,
    blank_id: int,
    enhanced: bool = True,
) -> ParaformerV2Config:
    config_cls = BetterParaformerV2Config if enhanced else ParaformerV2Config
    return config_cls.from_variant(
        _paraformer_variant_name(name),
        input_dim=input_dim,
        vocab_size=vocab_size,
        blank_id=blank_id,
    )


def paraformer_config_from_mapping(values: Mapping[str, object]) -> ParaformerV2Config:
    architecture = str(values.get("architecture", "paraformer"))
    config_cls = (
        BetterParaformerV2Config if architecture == "paraformer_better" else ParaformerV2Config
    )
    valid_fields = set(config_cls.__dataclass_fields__)
    return config_cls(**{key: value for key, value in values.items() if key in valid_fields})


class ParaformerASR(nn.Module):
    def __init__(
        self,
        encoder_config: ParaformerV2Config,
        *,
        alignment_mode: str = "viterbi",
        alignment_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.alignment_mode = alignment_mode
        self.alignment_backend = alignment_backend
        self.aed_decoder = None
        self.model = (
            BetterParaformerV2(encoder_config)
            if isinstance(encoder_config, BetterParaformerV2Config)
            else ParaformerV2(encoder_config)
        )

    def _ctc_loss(
        self,
        log_probs: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        blank_id: int,
    ) -> Tensor:
        per_sample_losses = F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            output_lengths,
            target_lengths,
            blank=blank_id,
            reduction="none",
            zero_infinity=True,
        )
        return (per_sample_losses / target_lengths.clamp_min(1)).mean()

    def _loss_from_outputs(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        target_lengths: Tensor,
    ) -> dict[str, Tensor]:
        ctc_loss = F.ctc_loss(
            outputs["ctc_log_probs"].transpose(0, 1),
            targets,
            outputs["encoder_lengths"],
            target_lengths,
            blank=self.encoder_config.resolved_blank_id,
            zero_infinity=True,
        )
        if isinstance(self.model, BetterParaformerV2):
            shallow_ctc_loss = F.ctc_loss(
                outputs["shallow_ctc_log_probs"].transpose(0, 1),
                targets,
                outputs["encoder_lengths"],
                target_lengths,
                blank=self.encoder_config.resolved_blank_id,
                zero_infinity=True,
            )
            base_ce_loss = better_masked_cross_entropy(
                outputs["initial_decoder_logits"],
                targets,
                target_lengths,
            )
            refined_ce_loss = better_masked_cross_entropy(
                outputs["decoder_logits"],
                targets,
                target_lengths,
            )
            boundary_targets = build_boundary_targets(
                outputs["alignments"],
                outputs["encoder_lengths"],
            )
            boundary_mask = ~lengths_to_padding_mask(
                outputs["encoder_lengths"],
                outputs["boundary_logits"].size(1),
            )
            boundary_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"][boundary_mask],
                boundary_targets[boundary_mask],
            )
            total = (
                ctc_loss
                + self.encoder_config.shallow_ctc_loss_weight * shallow_ctc_loss
                + base_ce_loss
                + self.encoder_config.refinement_loss_weight * refined_ce_loss
                + self.encoder_config.boundary_loss_weight * boundary_loss
            )
            return {
                "loss": total,
                "ctc_loss": ctc_loss.detach(),
                "shallow_ctc_loss": shallow_ctc_loss.detach(),
                "ce_loss": refined_ce_loss.detach(),
                "boundary_loss": boundary_loss.detach(),
            }

        ce_loss = masked_cross_entropy(outputs["decoder_logits"], targets, target_lengths)
        return {
            "loss": ctc_loss + ce_loss,
            "ctc_loss": ctc_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        return_training_outputs: bool = False,
        targets: Tensor | None = None,
        target_lengths: Tensor | None = None,
        blank_id: int | None = None,
        return_main_log_probs: bool = False,
        decoder_inputs: Tensor | None = None,
        liberta_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | dict[str, Any]:
        del liberta_lengths
        if decoder_inputs is not None:
            raise RuntimeError("AED decoder is not supported by the Paraformer training path.")
        if blank_id is not None and blank_id != self.encoder_config.resolved_blank_id:
            raise ValueError(
                "Paraformer checkpoint blank id does not match the tokenizer blank id: "
                f"{self.encoder_config.resolved_blank_id} != {blank_id}."
            )

        outputs = self.model(
            features,
            feature_lengths,
            targets,
            target_lengths,
            alignment_mode=self.alignment_mode,
            alignment_backend=self.alignment_backend,
        )
        losses = (
            self._loss_from_outputs(outputs, targets, target_lengths)
            if return_training_outputs and targets is not None and target_lengths is not None
            else None
        )

        log_probs = outputs["ctc_log_probs"]
        output_lengths = outputs["encoder_lengths"]
        if not return_training_outputs:
            return log_probs.exp(), output_lengths

        result: dict[str, Any] = {
            "encoded": outputs.get("decoder_logits", log_probs),
            "output_lengths": output_lengths,
            "main_ctc_loss": losses["loss"] if losses is not None else None,
        }
        if return_main_log_probs:
            result["main_log_probs"] = log_probs
            result["main_logits"] = log_probs
        return result

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(features, feature_lengths)
        return outputs["ctc_log_probs"], outputs["encoder_lengths"]

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)
