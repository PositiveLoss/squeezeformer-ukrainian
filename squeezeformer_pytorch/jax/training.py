from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.core import FrozenDict
from flax.training import train_state


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any]


def _lengths_to_paddings(lengths: jnp.ndarray, max_length: int) -> jnp.ndarray:
    positions = jnp.arange(max_length, dtype=jnp.int32)
    mask = positions[None, :] < lengths[:, None]
    return (~mask).astype(jnp.float32)


def ctc_loss(
    logits: jnp.ndarray,
    logit_lengths: jnp.ndarray,
    labels: jnp.ndarray,
    label_lengths: jnp.ndarray,
    *,
    blank_id: int = 0,
) -> jnp.ndarray:
    logit_paddings = _lengths_to_paddings(logit_lengths, logits.shape[1])
    label_paddings = _lengths_to_paddings(label_lengths, labels.shape[1])
    per_example = optax.ctc_loss(
        logits=logits,
        logit_paddings=logit_paddings,
        labels=labels,
        label_paddings=label_paddings,
        blank_id=blank_id,
    )
    return jnp.mean(per_example)


def create_adamw(
    learning_rate: float,
    *,
    weight_decay: float = 1e-2,
    grad_clip_norm: float | None = 1.0,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))
    transforms.append(optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay))
    return optax.chain(*transforms)


def create_train_state(
    rng: jax.Array,
    model: Any,
    *,
    example_features: jnp.ndarray,
    example_feature_lengths: jnp.ndarray,
    tx: optax.GradientTransformation,
) -> TrainState:
    variables = model.init(
        {"params": rng, "dropout": rng},
        example_features,
        example_feature_lengths,
        deterministic=False,
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables.get("batch_stats", FrozenDict()),
    )


def train_step(
    state: TrainState,
    batch: dict[str, jnp.ndarray],
    *,
    dropout_rng: jax.Array,
    blank_id: int = 0,
    axis_name: str | None = None,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    def loss_fn(
        params: FrozenDict[str, Any],
    ) -> tuple[jnp.ndarray, tuple[FrozenDict[str, Any], dict[str, jnp.ndarray]]]:
        variables = {"params": params, "batch_stats": state.batch_stats}
        (logits, output_lengths), updates = state.apply_fn(
            variables,
            batch["features"],
            batch["feature_lengths"],
            deterministic=False,
            rngs={"dropout": dropout_rng},
            mutable=["batch_stats"],
        )
        loss = ctc_loss(
            logits,
            output_lengths,
            batch["labels"],
            batch["label_lengths"],
            blank_id=blank_id,
        )
        metrics = {
            "loss": loss,
            "frames": jnp.sum(output_lengths),
            "labels": jnp.sum(batch["label_lengths"]),
        }
        return loss, (updates["batch_stats"], metrics)

    (loss, (new_batch_stats, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    metrics["loss"] = loss
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)
        metrics = jax.lax.pmean(metrics, axis_name=axis_name)
    new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    return new_state, metrics


def eval_step(
    state: TrainState,
    batch: dict[str, jnp.ndarray],
    *,
    blank_id: int = 0,
    axis_name: str | None = None,
) -> dict[str, jnp.ndarray]:
    logits, output_lengths = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch["features"],
        batch["feature_lengths"],
        deterministic=True,
    )
    metrics = {
        "loss": ctc_loss(
            logits,
            output_lengths,
            batch["labels"],
            batch["label_lengths"],
            blank_id=blank_id,
        ),
        "frames": jnp.sum(output_lengths),
        "labels": jnp.sum(batch["label_lengths"]),
    }
    if axis_name is not None:
        metrics = jax.lax.pmean(metrics, axis_name=axis_name)
    return metrics


def replicate_state(state: TrainState) -> TrainState:
    return jax_utils.replicate(state)


def unreplicate_state(state: TrainState) -> TrainState:
    return jax_utils.unreplicate(state)


def shard_batch(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    device_count = jax.local_device_count()

    def _reshape(x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[0] % device_count != 0:
            raise ValueError(
                f"Batch size {x.shape[0]} must be divisible by local_device_count {device_count}."
            )
        return x.reshape((device_count, x.shape[0] // device_count, *x.shape[1:]))

    return jax.tree_util.tree_map(_reshape, batch)
