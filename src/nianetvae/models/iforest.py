"""Isolation Forest runtime without detector-owned preprocessing."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestRuntime:
    name = "iforest"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        contamination: str | float = "auto",
        seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.model: IsolationForest | None = None

    def fit(self, frame: pd.DataFrame) -> IsolationForestRuntime:
        if frame is None or len(frame) < 2:
            raise ValueError("Isolation Forest requires at least two training rows.")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(frame.to_numpy(dtype=float, copy=False))
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Isolation Forest must be fitted before scoring.")
        values = -self.model.decision_function(frame.to_numpy(dtype=float, copy=False))
        return pd.Series(values, index=frame.index, name="anomaly_score", dtype=float)

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted Isolation Forest.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "seed": self.seed,
                "model": self.model,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> IsolationForestRuntime:
        payload = joblib.load(path)
        runtime = cls(
            n_estimators=payload["n_estimators"],
            contamination=payload["contamination"],
            seed=payload["seed"],
        )
        runtime.model = payload["model"]
        return runtime
