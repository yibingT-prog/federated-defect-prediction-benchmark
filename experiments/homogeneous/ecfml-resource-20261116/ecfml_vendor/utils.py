from __future__ import annotations

import random
from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def detach_state(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in state.items())


def average_tensor_dicts(dicts: list[Mapping[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    if not dicts:
        raise ValueError("Cannot average an empty list of tensor dictionaries.")
    names = list(dicts[0].keys())
    return OrderedDict((name, torch.stack([item[name] for item in dicts], dim=0).mean(dim=0)) for name in names)
