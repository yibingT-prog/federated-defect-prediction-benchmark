from __future__ import annotations

from .dataset import load_project
from .preprocessing import PreparedProject, prepare_project


def prepare_all_projects(config: dict) -> dict[str, PreparedProject]:
    prepared = {}
    for name in config["data"]["projects"]:
        project = load_project(
            config["data"]["raw_dir"],
            name,
            config["data"].get("label_column"),
            config["data"].get("non_feature_columns"),
        )
        prepared[name] = prepare_project(
            project,
            test_size=float(config["data"]["test_size"]),
            seed=int(config["training"]["seed"]),
            scaling=str(config["data"]["scaling"]),
            missing_column_threshold=float(config["data"]["missing_column_threshold"]),
        )
    return prepared
