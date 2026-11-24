"""Client-side global-L2 clipping and Gaussian perturbation for native TFF."""
from __future__ import annotations

from dataclasses import dataclass
import collections
import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    clip: bool
    sigma: float
    clip_norm: float = 1.0


CONDITIONS = {
    "none": Condition("none", False, 0.0),
    "clip_only": Condition("clip_only", True, 0.0),
    "sigma_0p1": Condition("sigma_0p1", True, 0.1),
    "sigma_0p25": Condition("sigma_0p25", True, 0.25),
    "sigma_0p5": Condition("sigma_0p5", True, 0.5),
    "sigma_1p0": Condition("sigma_1p0", True, 1.0),
}


def perturb_flat(update, condition: Condition, rng: np.random.Generator):
    value = np.asarray(update, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("update contains non-finite values")
    if condition.name == "none":
        return value.copy(), {"pre_norm": float(np.linalg.norm(value)), "clip_scale": 1.0,
                              "post_clip_norm": float(np.linalg.norm(value)), "noise_norm": 0.0}
    if condition.clip_norm <= 0 or condition.sigma < 0:
        raise ValueError("invalid clipping or noise parameter")
    pre = float(np.linalg.norm(value))
    scale = min(1.0, condition.clip_norm / (pre + 1e-12))
    clipped = value * scale
    noise = rng.normal(0.0, condition.sigma * condition.clip_norm, size=value.size)
    result = clipped + noise
    return result, {"pre_norm": pre, "clip_scale": scale,
                    "post_clip_norm": float(np.linalg.norm(clipped)),
                    "noise_norm": float(np.linalg.norm(noise))}


def condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown privacy condition: {name}") from exc


class ClientUpdateAggregationFactory:
    """Perturb each complete client update before the weighted mean."""

    def __new__(cls, tff, cond: Condition, clients_per_round: int):
        from tensorflow_federated.python.aggregators import factory
        class _Factory(ClientUpdateAggregationFactory, factory.WeightedAggregationFactory):
            pass
        obj = object.__new__(_Factory)
        obj.tff = tff; obj.cond = cond; obj.clients_per_round = clients_per_round
        return obj

    def __init__(self, tff, cond: Condition, clients_per_round: int):
        self.tff = tff
        self.cond = cond
        self.clients_per_round = clients_per_round

    def create(self, value_type, weight_type):
        tff = self.tff
        inner = tff.aggregators.MeanFactory().create(value_type, weight_type)
        if self.cond.name == "none":
            return inner

        import tensorflow as tf
        from tensorflow_federated.python.core.impl.types import computation_types
        from tensorflow_federated.python.core.impl.types import placements
        from tensorflow_federated.python.core.templates import aggregation_process
        from tensorflow_federated.python.core.templates import measured_process

        @tff.tf_computation(value_type)
        def perturb(value):
            pieces = tf.nest.flatten(value)
            flat = tf.concat([tf.reshape(tf.cast(x, tf.float32), [-1]) for x in pieces], axis=0)
            norm = tf.linalg.norm(flat)
            scale = tf.minimum(1.0, tf.cast(self.cond.clip_norm, tf.float32) / (norm + 1e-12))
            clipped = flat * scale
            noise = tf.random.normal(tf.shape(flat), stddev=self.cond.sigma * self.cond.clip_norm,
                                     dtype=tf.float32)
            noisy = clipped + noise
            sizes = [tf.size(x) for x in pieces]
            splits = tf.split(noisy, sizes)
            rebuilt = [tf.reshape(tf.cast(part, original.dtype), tf.shape(original))
                       for part, original in zip(splits, pieces)]
            result = tf.nest.pack_sequence_as(value, rebuilt)
            post_norm = tf.linalg.norm(clipped)
            noise_norm = tf.linalg.norm(noise)
            return collections.OrderedDict(value=result, diagnostics=collections.OrderedDict(
                pre_norm=norm, post_clip_norm=post_norm, noise_norm=noise_norm,
                clip_scale=scale, was_clipped=tf.cast(norm > self.cond.clip_norm, tf.float32)))

        mapped_type = perturb.type_signature.result

        @tff.tf_computation(mapped_type)
        def take_value(mapped):
            return mapped["value"]

        @tff.tf_computation(mapped_type)
        def take_diagnostics(mapped):
            return mapped["diagnostics"]

        @tff.federated_computation()
        def initialize_fn():
            return inner.initialize()

        @tff.federated_computation(
            initialize_fn.type_signature.result,
            computation_types.FederatedType(value_type, placements.CLIENTS),
            computation_types.FederatedType(weight_type, placements.CLIENTS))
        def next_fn(state, value, weight):
            perturbed = tff.federated_map(perturb, value)
            client_values = tff.federated_map(take_value, perturbed)
            diag = tff.federated_sum(tff.federated_map(take_diagnostics, perturbed))
            result = inner.next(state, client_values, weight)

            @tff.tf_computation(diag.type_signature.member)
            def averages(value):
                # Build the divisor inside the TensorFlow computation.  A
                # module-level eager tensor cannot be captured by TFF 0.20.
                return tf.nest.map_structure(
                    lambda x: x / tf.cast(self.clients_per_round, x.dtype), value)

            measurements = tff.federated_map(averages, diag)
            return measured_process.MeasuredProcessOutput(
                state=result.state, result=result.result,
                measurements=tff.federated_zip(collections.OrderedDict(
                    inner=result.measurements, client_update=measurements)))

        return aggregation_process.AggregationProcess(initialize_fn, next_fn)
