from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


def privatize_updates(
    updates: Mapping[str, torch.Tensor],
    clipping_bound: float,
    epsilon: float,
    enabled: bool = True,
    clipping_mode: str = "per_coordinate",
) -> OrderedDict[str, torch.Tensor]:
    if not enabled:
        return OrderedDict((name, value.detach().clone()) for name, value in updates.items())
    if epsilon <= 0:
        raise ValueError("epsilon must be positive when differential privacy is enabled.")
    if clipping_bound <= 0:
        raise ValueError("clipping_bound must be positive.")

    scale = clipping_bound / epsilon
    private = OrderedDict()
    for name, tensor in updates.items():
        clipped = clip_tensor(tensor.detach(), clipping_bound, clipping_mode)
        noise = torch.distributions.Laplace(
            loc=torch.zeros_like(clipped),
            scale=torch.full_like(clipped, scale),
        ).sample()
        private[name] = clipped + noise
    return private


def clip_tensor(tensor: torch.Tensor, clipping_bound: float, clipping_mode: str) -> torch.Tensor:
    if clipping_mode == "per_coordinate":
        return tensor.clamp(min=-clipping_bound, max=clipping_bound)
    if clipping_mode == "l1":
        norm = tensor.abs().sum()
        factor = min(1.0, clipping_bound / (norm.item() + 1e-12))
        return tensor * factor
    if clipping_mode == "l2":
        norm = tensor.norm(p=2)
        factor = min(1.0, clipping_bound / (norm.item() + 1e-12))
        return tensor * factor
    raise ValueError("clipping_mode must be 'per_coordinate', 'l1', or 'l2'.")
