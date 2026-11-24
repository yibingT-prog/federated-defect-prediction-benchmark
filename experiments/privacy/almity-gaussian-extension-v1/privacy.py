"""Client-update clipping and Gaussian perturbation for the ALMITY extension."""
from __future__ import annotations

from dataclasses import dataclass
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


def perturb_update(dw, db, condition: Condition, rng: np.random.Generator):
    """Perturb one complete client update, including the scalar bias."""
    dw = np.asarray(dw)
    db = np.asarray(db).reshape(())
    if not np.isfinite(dw).all() or not np.isfinite(db):
        raise ValueError("update contains non-finite values")
    if condition.name == "none":
        return dw.copy(), db.copy(), {"pre_norm": float(np.linalg.norm(np.r_[dw.ravel(), db]))}
    if condition.clip_norm <= 0 or condition.sigma < 0:
        raise ValueError("invalid clipping or noise parameter")
    flat = np.concatenate([dw.ravel(), np.asarray([db], dtype=dw.dtype)])
    pre = float(np.linalg.norm(flat))
    scale = min(1.0, condition.clip_norm / (pre + 1e-12))
    clipped = flat * scale
    if condition.sigma:
        clipped = clipped + rng.normal(0.0, condition.sigma * condition.clip_norm, size=flat.size).astype(dw.dtype)
    out_w = clipped[:-1].reshape(dw.shape)
    out_b = np.asarray(clipped[-1], dtype=dw.dtype).reshape(())
    return out_w, out_b, {
        "pre_norm": pre,
        "clip_scale": scale,
        "post_clip_norm": float(np.linalg.norm(flat * scale)),
        "noise_norm": float(np.linalg.norm(clipped - flat * scale)),
    }


def condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown privacy condition: {name}") from exc
