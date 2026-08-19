"""Maintenance-risk construction and deterministic operating-point selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import EvaluationConfig
from ..dataloaders.sequences import true_runs
from .event import event_scores


def parse_theta_grid(specification: str) -> list[float]:
    parts = [part.strip() for part in str(specification).split(":")]
    if len(parts) != 3:
        raise ValueError("theta_grid must use start:stop:step syntax.")
    start, stop, step = (float(part) for part in parts)
    if step <= 0.0 or stop < start:
        raise ValueError("Invalid theta grid bounds.")
    count = int(np.floor((stop - start) / step + 1e-12))
    return [round(start + offset * step, 12) for offset in range(count + 1)]


def theta_values(config: EvaluationConfig) -> list[float]:
    return sorted(
        {*parse_theta_grid(config.theta_grid), *(float(value) for value in config.extra_thresholds)}
    )


def build_segmented_maintenance_risk(
    risk_score: pd.Series,
    segment_masks: Sequence[pd.Series],
    *,
    exceedance_quantile: float,
    risk_window_minutes: int,
) -> pd.Series:
    if not 0.0 < float(exceedance_quantile) <= 1.0:
        raise ValueError("exceedance_quantile must be in (0,1].")
    if int(risk_window_minutes) < 1:
        raise ValueError("risk_window_minutes must be positive.")
    if risk_score.index.has_duplicates or not risk_score.index.is_monotonic_increasing:
        raise ValueError("Risk-score timestamps must be unique and monotonically increasing.")
    output = pd.Series(np.nan, index=risk_score.index, name="maintenance_risk", dtype=float)
    for segment_mask in segment_masks:
        mask = (
            segment_mask.reindex(risk_score.index).fillna(False).astype(bool) & risk_score.notna()
        )
        # A rolling risk value must never bridge an excluded row or telemetry
        # gap.  The same gap rule is used by the recurrent sequence builder.
        for positions in true_runs(mask):
            values = risk_score.iloc[positions]
            exceedance = (values >= float(exceedance_quantile)).astype(float)
            rolled = exceedance.rolling(f"{int(risk_window_minutes)}min", min_periods=1).mean()
            output.loc[rolled.index] = rolled
    return output


def evaluate_risk_thresholds(
    risk: pd.Series,
    maintenance_windows: Sequence,
    *,
    eval_mask: pd.Series,
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluation = eval_mask.reindex(risk.index).fillna(False).astype(bool) & risk.notna()
    for theta in theta_values(config):
        prediction = (risk >= theta) & evaluation
        row = event_scores(prediction, maintenance_windows, config.fixed_lead_minutes)
        total = int(evaluation.sum())
        alarms = int((prediction & evaluation).sum())
        coverage = alarms / total if total else 0.0
        row.update(
            {
                "maintenance_risk_theta": theta,
                "threshold": theta,
                "coverage": coverage,
                "coverage_percent": coverage * 100.0,
                "alarm_points": alarms,
                "total_points": total,
                "target_gap": max(0.0, config.target_recall - row["recall"])
                + max(0.0, coverage - config.target_coverage),
            }
        )
        rows.append(row)
    return rows


def select_operating_point(
    rows: Sequence[dict[str, Any]], config: EvaluationConfig
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select an operating point from an empty theta sweep.")
    feasible = [
        row
        for row in rows
        if float(row["recall"]) >= config.target_recall
        and float(row["coverage"]) < config.target_coverage
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                float(row["recall"]),
                -float(row["threshold"]),
            ),
        )
        mode = "feasible"
    else:
        selected = min(
            rows,
            key=lambda row: (
                float(row["target_gap"]),
                -float(row["f1"]),
                -float(row["precision"]),
                -float(row["recall"]),
                float(row["threshold"]),
            ),
        )
        mode = "fallback"
    result = dict(selected)
    result.update(
        {
            "selection_mode": mode,
            "target_recall": config.target_recall,
            "target_coverage": config.target_coverage,
            "selection_scope": config.selection_scope,
        }
    )
    return result
