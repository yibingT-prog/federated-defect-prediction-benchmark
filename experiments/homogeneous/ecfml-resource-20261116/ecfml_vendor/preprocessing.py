from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .dataset import ProjectData


@dataclass
class PreparedProject:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class ProjectPreprocessor:
    def __init__(self, scaling: str = "standard", missing_column_threshold: float = 0.5) -> None:
        self.scaling = scaling
        self.missing_column_threshold = missing_column_threshold
        self.kept_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_fill: dict[str, float] = {}
        self.categorical_fill: dict[str, str] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.scaler: StandardScaler | MinMaxScaler | None = None

    def fit(self, frame: pd.DataFrame) -> "ProjectPreprocessor":
        missing_fraction = frame.isna().mean()
        self.kept_columns = [column for column in frame.columns if missing_fraction[column] <= self.missing_column_threshold]
        data = frame[self.kept_columns].copy()
        self.numeric_columns = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
        self.categorical_columns = [column for column in data.columns if column not in self.numeric_columns]

        for column in self.numeric_columns:
            self.numeric_fill[column] = float(pd.to_numeric(data[column], errors="coerce").median())
        for column in self.categorical_columns:
            mode = data[column].astype(str).mode(dropna=True)
            self.categorical_fill[column] = str(mode.iloc[0]) if len(mode) else ""
            categories = sorted(data[column].fillna(self.categorical_fill[column]).astype(str).unique())
            self.category_maps[column] = {value: index for index, value in enumerate(categories)}

        encoded = self._encode(data)
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling must be either 'standard' or 'minmax'.")
        self.scaler.fit(encoded)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        data = frame.reindex(columns=self.kept_columns).copy()
        encoded = self._encode(data)
        return self.scaler.transform(encoded).astype(np.float32)

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(index=frame.index)
        for column in self.numeric_columns:
            data[column] = pd.to_numeric(frame[column], errors="coerce").fillna(self.numeric_fill[column])
        for column in self.categorical_columns:
            fill = self.categorical_fill[column]
            mapping = self.category_maps[column]
            data[column] = frame[column].fillna(fill).astype(str).map(lambda value: mapping.get(value, -1))
        return data[self.numeric_columns + self.categorical_columns]


def prepare_project(project: ProjectData, test_size: float, seed: int, scaling: str, missing_column_threshold: float) -> PreparedProject:
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        project.features,
        project.labels.to_numpy(dtype=np.int64),
        test_size=test_size,
        random_state=seed,
        stratify=project.labels,
    )
    preprocessor = ProjectPreprocessor(scaling=scaling, missing_column_threshold=missing_column_threshold).fit(x_train_raw)
    x_train = preprocessor.transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)
    return PreparedProject(
        name=project.name,
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_test=x_test,
        y_test=y_test.astype(np.int64),
    )
