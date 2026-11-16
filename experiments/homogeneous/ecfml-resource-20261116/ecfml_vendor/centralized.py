from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .baselines import build_baseline
from .metrics import choose_threshold, classification_metrics
from .preprocessing import PreparedProject
from .training import build_criterion, iter_batches, predict_probabilities
from .utils import resolve_device, set_seed


@dataclass
class BaselineRun:
    target: str
    model_name: str
    activation: str
    metrics: dict[str, float]
    model: nn.Module


def train_baseline(projects: dict[str, PreparedProject], target: str, model_name: str, activation: str, config: dict) -> BaselineRun:
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["training"]["device"])
    model = build_baseline(
        model_name,
        input_dim=int(config["model"]["input_dim"]),
        num_classes=int(config["model"]["num_classes"]),
        activation=activation,
    ).to(device)
    source_names = [name for name in projects if config["federated"]["train_on_target_project"] or name != target]
    x_train = np.concatenate([projects[name].x_train for name in source_names], axis=0)
    y_train = np.concatenate([projects[name].y_train for name in source_names], axis=0)
    criterion = build_criterion(y_train, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    model.train()
    rng = np.random.default_rng(int(config["training"]["seed"]))
    batch_size = int(config["training"]["batch_size"])
    for _ in range(int(config["training"]["epochs"])):
        order = rng.permutation(len(y_train))
        x_epoch = x_train[order]
        y_epoch = y_train[order]
        for x_batch, y_batch in iter_batches(x_epoch, y_epoch, batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    train_probabilities = predict_probabilities(model, x_train, y_train, device)
    threshold = choose_threshold(
        y_train,
        train_probabilities,
        str(config.get("evaluation", {}).get("threshold_strategy", "fixed_0.5")),
    )
    test_project = projects[target]
    test_probabilities = predict_probabilities(model, test_project.x_test, test_project.y_test, device)
    metrics = classification_metrics(test_project.y_test, test_probabilities, threshold)
    return BaselineRun(target=target, model_name=model_name, activation=activation, metrics=metrics, model=model)
