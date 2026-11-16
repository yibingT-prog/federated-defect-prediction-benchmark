from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore

from .differential_privacy import privatize_updates
from .metrics import choose_threshold, classification_metrics
from .model import build_model
from .preprocessing import PreparedProject
from .utils import average_tensor_dicts, resolve_device, set_seed


@dataclass
class TrainRun:
    target: str
    metrics: dict[str, float]
    history: list[dict[str, float]]
    model: nn.Module


def train_ecfml(projects: dict[str, PreparedProject], target: str, config: dict) -> TrainRun:
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = resolve_device(config["training"]["device"])
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    rng = np.random.default_rng(seed)

    source_names = [name for name in projects if config["federated"]["train_on_target_project"] or name != target]
    if not source_names:
        raise ValueError("No source clients available for federated meta-training.")
    privacy_enabled = bool(config["privacy"]["enabled"])
    clipping_bound = config["privacy"].get("clipping_bound")
    if privacy_enabled and clipping_bound is None:
        raise ValueError(
            "privacy.clipping_bound is required for Laplace DP. "
            "The paper does not report C, so set it explicitly in configs/paper.yaml "
            "or pass --clipping-bound to the training script."
        )
    clipping_bound_value = float(clipping_bound) if clipping_bound is not None else 1.0

    history: list[dict[str, float]] = []
    rounds = int(config["federated"].get("communication_rounds", config["training"]["epochs"]))
    for round_index in range(rounds):
        selected_names = sample_clients(source_names, float(config["federated"]["client_sampling_fraction"]), rng)
        private_updates = []
        losses = []
        for name in selected_names:
            update_dict, loss_value = first_order_client_update(model, projects[name], config, rng, device)
            private_updates.append(
                privatize_updates(
                    update_dict,
                    clipping_bound=clipping_bound_value,
                    epsilon=float(config["privacy"]["epsilon"]),
                    enabled=privacy_enabled,
                    clipping_mode=str(config["privacy"]["clipping_mode"]),
                )
            )
            losses.append(loss_value)

        averaged = average_tensor_dicts(private_updates)
        aggregation_mode = str(config["federated"].get("aggregation_mode", "meta_gradient"))
        if aggregation_mode == "meta_gradient":
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                parameter.grad = averaged[name].to(device)
            optimizer.step()
        elif aggregation_mode == "client_delta":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.add_(averaged[name].to(device))
        else:
            raise ValueError("federated.aggregation_mode must be 'meta_gradient' or 'client_delta'.")
        history.append({"round": float(round_index + 1), "query_loss": float(np.mean(losses))})

    adapted_model = adapt_to_target(model, projects[target], config, device)
    train_probabilities = predict_probabilities(adapted_model, projects[target].x_train, projects[target].y_train, device)
    threshold = choose_threshold(
        projects[target].y_train,
        train_probabilities,
        str(config.get("evaluation", {}).get("threshold_strategy", "fixed_0.5")),
    )
    metrics = evaluate(adapted_model, projects[target].x_test, projects[target].y_test, device, threshold)
    return TrainRun(target=target, metrics=metrics, history=history, model=adapted_model)


def sample_clients(names: list[str], fraction: float, rng: np.random.Generator) -> list[str]:
    count = max(1, int(np.ceil(len(names) * fraction)))
    if count >= len(names):
        return names
    indices = rng.choice(len(names), size=count, replace=False)
    return [names[int(index)] for index in indices]


def first_order_client_update(
    model: nn.Module,
    project: PreparedProject,
    config: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    model.train()
    support_x, support_y, query_x, query_y = support_query_batch(project, config, rng, device)
    criterion = build_criterion(project.y_train, config, device)
    params = OrderedDict((name, parameter) for name, parameter in model.named_parameters())

    support_logits = model(support_x)
    support_loss = criterion(support_logits, support_y)
    support_grads = torch.autograd.grad(support_loss, params.values(), create_graph=False, allow_unused=False)
    adapted = OrderedDict(
        (name, parameter - float(config["training"]["inner_learning_rate"]) * grad.detach())
        for (name, parameter), grad in zip(params.items(), support_grads)
    )
    aggregation_mode = str(config["federated"].get("aggregation_mode", "meta_gradient"))
    if aggregation_mode == "client_delta":
        delta = OrderedDict((name, adapted[name].detach().clone() - parameter.detach().clone()) for name, parameter in params.items())
        return delta, float(support_loss.detach().cpu().item())

    query_logits = call_model_with_params(model, adapted, query_x)
    query_loss = criterion(query_logits, query_y)
    query_grads = torch.autograd.grad(query_loss, adapted.values(), create_graph=False, allow_unused=False)
    grad_dict = OrderedDict((name, grad.detach().clone()) for name, grad in zip(adapted.keys(), query_grads))
    return grad_dict, float(query_loss.detach().cpu().item())


def call_model_with_params(model: nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    state = OrderedDict((name, tensor) for name, tensor in model.named_buffers())
    state.update(params)
    return functional_call(model, state, (x,))


def support_query_batch(
    project: PreparedProject,
    config: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(config["training"]["batch_size"])
    total = max(2, batch_size * 2)
    sampling = str(config["training"].get("support_query_sampling", "random"))
    if sampling == "random":
        replace = len(project.y_train) < total
        indices = rng.choice(len(project.y_train), size=total, replace=replace)
    elif sampling == "balanced":
        indices = balanced_indices(project.y_train, total, rng)
    else:
        raise ValueError("training.support_query_sampling must be 'random' or 'balanced'.")
    split = max(1, min(total - 1, int(total * float(config["training"]["support_query_split"]))))
    support_idx = indices[:split]
    query_idx = indices[split:]
    return (
        torch.as_tensor(project.x_train[support_idx], dtype=torch.float32, device=device),
        torch.as_tensor(project.y_train[support_idx], dtype=torch.long, device=device),
        torch.as_tensor(project.x_train[query_idx], dtype=torch.float32, device=device),
        torch.as_tensor(project.y_train[query_idx], dtype=torch.long, device=device),
    )


def balanced_indices(y: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) == 0 or len(negative) == 0:
        replace = len(y) < total
        return rng.choice(len(y), size=total, replace=replace)
    positive_count = total // 2
    negative_count = total - positive_count
    positive_idx = rng.choice(positive, size=positive_count, replace=len(positive) < positive_count)
    negative_idx = rng.choice(negative, size=negative_count, replace=len(negative) < negative_count)
    indices = np.concatenate([positive_idx, negative_idx])
    rng.shuffle(indices)
    return indices


def adapt_to_target(model: nn.Module, project: PreparedProject, config: dict, device: torch.device) -> nn.Module:
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=float(config["training"]["inner_learning_rate"]))
    criterion = build_criterion(project.y_train, config, device)
    steps = int(config["training"]["target_adaptation_steps"])
    for _ in range(steps):
        for x_batch, y_batch in iter_batches(project.x_train, project.y_train, int(config["training"]["batch_size"]), device):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(adapted(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return adapted


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(y), batch_size):
        end = start + batch_size
        yield (
            torch.as_tensor(x[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(y[start:end], dtype=torch.long, device=device),
        )


def build_criterion(y_train: np.ndarray, config: dict, device: torch.device) -> nn.Module:
    weighting = str(config["training"].get("class_weighting", "none"))
    if weighting == "none":
        return nn.CrossEntropyLoss()
    if weighting != "balanced":
        raise ValueError("training.class_weighting must be 'none' or 'balanced'.")
    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    probabilities = predict_probabilities(model, x, y, device)
    return classification_metrics(y, probabilities, threshold)


def predict_probabilities(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x_batch, _ in iter_batches(x, y, 512, device):
            probabilities.append(torch.softmax(model(x_batch), dim=-1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)
