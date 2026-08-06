"""One immutable, auditable preprocessing implementation for every workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenPreprocessor:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    passthrough_indices: tuple[int, ...]
    fitted_row_count: int
    policy: str = "binary_passthrough_v1"
    schema_version: str = "1.0"

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        binary_feature_names: Sequence[str],
        policy: str = "binary_passthrough_v1",
    ) -> FrozenPreprocessor:
        if policy != "binary_passthrough_v1":
            raise ValueError(f"Unsupported preprocessing policy={policy!r}.")
        if frame is None or frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty frame.")
        values = frame.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessor fit frame contains non-finite values.")
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
        feature_names = tuple(str(column) for column in frame.columns)
        binary = {str(name) for name in binary_feature_names}
        passthrough = tuple(
            index
            for index, feature_name in enumerate(feature_names)
            if feature_name.split("__", 1)[0] in binary
        )
        if not passthrough:
            raise ValueError("binary_passthrough_v1 matched no engineered features.")
        means = means.astype(np.float64, copy=True)
        scales = scales.astype(np.float64, copy=True)
        means[list(passthrough)] = 0.0
        scales[list(passthrough)] = 1.0
        means.setflags(write=False)
        scales.setflags(write=False)
        return cls(
            feature_names=feature_names,
            means=means,
            scales=scales,
            passthrough_indices=passthrough,
            fitted_row_count=int(len(frame)),
            policy=policy,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in frame.columns) != self.feature_names:
            raise ValueError("Feature names/order do not match the frozen preprocessing contract.")
        values = frame.to_numpy(dtype=np.float64, copy=False)
        transformed = (values - self.means) / self.scales
        return pd.DataFrame(
            transformed.astype(np.float32, copy=False),
            index=frame.index,
            columns=frame.columns,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "passthrough_indices": list(self.passthrough_indices),
            "fitted_row_count": self.fitted_row_count,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> FrozenPreprocessor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("Unsupported frozen preprocessing schema.")
        means = np.asarray(payload["means"], dtype=np.float64)
        scales = np.asarray(payload["scales"], dtype=np.float64)
        means.setflags(write=False)
        scales.setflags(write=False)
        return cls(
            feature_names=tuple(payload["feature_names"]),
            means=means,
            scales=scales,
            passthrough_indices=tuple(int(value) for value in payload["passthrough_indices"]),
            fitted_row_count=int(payload["fitted_row_count"]),
            policy=str(payload["policy"]),
            schema_version=str(payload["schema_version"]),
        )
