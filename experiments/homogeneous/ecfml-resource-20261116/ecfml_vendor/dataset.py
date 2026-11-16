from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_CANDIDATES = {
    "bug",
    "bugs",
    "buggy",
    "defect",
    "defects",
    "defective",
    "label",
    "class",
    "is_defective",
    "numberofbugs",
    "number_of_bugs",
    "post",
}


@dataclass
class ProjectData:
    name: str
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str

    @property
    def features(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def labels(self) -> pd.Series:
        return self.frame[self.label_column]


def find_project_file(raw_dir: str | Path, project: str) -> Path:
    root = Path(raw_dir)
    patterns = [f"{project}.*", f"{project.lower()}.*", f"{project.upper()}.*"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = [path for path in candidates if path.suffix.lower() in {".csv", ".arff", ".txt"}]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {project}.csv or {project}.arff under {root.resolve()}. "
            "Place the AEEEM EQ/JDT/PDE files there first."
        )
    return sorted(candidates)[0]


def read_project_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".arff":
        from scipy.io import arff

        data, _ = arff.loadarff(path)
        frame = pd.DataFrame(data)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
        return frame
    return pd.read_csv(path)


def normalize_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def infer_label_column(frame: pd.DataFrame, configured: str | None = None) -> str:
    if configured:
        if configured not in frame.columns:
            raise ValueError(f"Configured label column {configured!r} does not exist.")
        return configured

    normalized = {column: normalize_column_name(column) for column in frame.columns}
    for column, name in normalized.items():
        compact = name.replace("_", "")
        if name in LABEL_CANDIDATES or compact in LABEL_CANDIDATES:
            return column

    last_column = frame.columns[-1]
    values = set(str(value).strip().lower() for value in frame[last_column].dropna().unique()[:10])
    if len(values) <= 5:
        return last_column
    raise ValueError("Could not infer the defect label column. Set data.label_column in configs/paper.yaml.")


def binarize_labels(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        return (pd.to_numeric(labels, errors="coerce").fillna(0) > 0).astype(int)

    positive = {"true", "yes", "y", "1", "buggy", "defective", "defect", "bug", "b"}
    return labels.astype(str).str.strip().str.lower().map(lambda value: 1 if value in positive else 0).astype(int)


def load_project(
    raw_dir: str | Path,
    project: str,
    label_column: str | None = None,
    non_feature_columns: list[str] | None = None,
) -> ProjectData:
    path = find_project_file(raw_dir, project)
    frame = read_project_file(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    label = infer_label_column(frame, label_column)
    frame[label] = binarize_labels(frame[label])
    ignored = set(non_feature_columns or [])
    features = [column for column in frame.columns if column != label and column not in ignored]
    return ProjectData(name=project, frame=frame, feature_columns=features, label_column=label)


def inspect_project(project: ProjectData, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = project.labels.astype(int)
    defects = int(labels.sum())
    clean = int(len(labels) - defects)
    stats = {
        "project": project.name,
        "samples": int(len(project.frame)),
        "features": int(len(project.feature_columns)),
        "defects": defects,
        "clean": clean,
        "imbalance_ratio": round(clean / defects, 3) if defects else np.inf,
        "missing_values": int(project.features.isna().sum().sum()),
        "categorical_columns": [
            column for column in project.feature_columns if not pd.api.types.is_numeric_dtype(project.frame[column])
        ],
        "label_column": project.label_column,
    }
    if expected:
        stats["matches_expected"] = {
            "samples": stats["samples"] == expected.get("samples"),
            "features": stats["features"] == expected.get("features"),
            "defects": stats["defects"] == expected.get("defects"),
        }
    return stats
