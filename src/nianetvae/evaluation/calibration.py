"""Shared empirical-CDF anomaly-score calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmpiricalCDFCalibrator:
    sorted_reference_scores: np.ndarray
    reference_timestamps: tuple[str, ...]
    method: str = "empirical_cdf_v1"

    @classmethod
    def fit(cls, scores: pd.Series) -> EmpiricalCDFCalibrator:
        if scores is None:
            raise ValueError("Calibration scores cannot be None.")
        finite = scores[np.isfinite(scores.to_numpy(dtype=float))].astype(float)
        if len(finite) < 2:
            raise ValueError("Calibration requires at least two finite scores.")
        values = np.sort(finite.to_numpy(dtype=np.float64, copy=True))
        values.setflags(write=False)
        timestamps = tuple(pd.Timestamp(value).isoformat() for value in finite.index)
        return cls(values, timestamps)

    def transform(self, scores: pd.Series) -> pd.Series:
        values = scores.to_numpy(dtype=float)
        finite = np.isfinite(values)
        risks = np.full(values.shape, np.nan, dtype=float)
        ranks = np.searchsorted(self.sorted_reference_scores, values[finite], side="right")
        risks[finite] = np.clip(ranks / float(len(self.sorted_reference_scores)), 0.0, 1.0)
        return pd.Series(risks, index=scores.index, name="risk_score")

    @property
    def reference_index_hash(self) -> str:
        payload = json.dumps(self.reference_timestamps, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.method.encode("utf-8"))
        digest.update(self.reference_index_hash.encode("ascii"))
        digest.update(self.sorted_reference_scores.tobytes())
        return digest.hexdigest()
