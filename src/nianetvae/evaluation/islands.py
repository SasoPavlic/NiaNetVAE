"""Alarm-island diagnostics for temporal localization and burden."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .event import alarm_intervals, interval_duration_minutes, normalize_windows


def analyze_alarm_islands(
    predictions: pd.Series,
    maintenance_windows: Sequence,
    *,
    early_warning_minutes: int,
    eval_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = eval_mask.reindex(predictions.index).fillna(False).astype(bool)
    active = predictions.reindex(mask.index).fillna(False).astype(bool) & mask
    events = normalize_windows(maintenance_windows)
    rows: list[dict[str, Any]] = []
    previous_end = None
    for island_id, (start, end) in enumerate(alarm_intervals(active), start=1):
        matching = []
        for event_index, (event_start, _event_end) in enumerate(events, start=1):
            warning_start = event_start - pd.Timedelta(minutes=int(early_warning_minutes))
            if start <= event_start and end >= warning_start:
                matching.append(event_index)
        points = int(active.loc[start:end].sum())
        rows.append(
            {
                "island_id": island_id,
                "start": start,
                "end": end,
                "duration_minutes": interval_duration_minutes(start, end, active.index),
                "point_count": points,
                "gap_from_previous_minutes": (
                    (start - previous_end).total_seconds() / 60.0
                    if previous_end is not None
                    else None
                ),
                "is_true_positive": bool(matching),
                "matched_event_ids": ",".join(str(value) for value in matching),
            }
        )
        previous_end = end
    frame = pd.DataFrame(rows)
    total = len(frame)
    true_count = int(frame["is_true_positive"].sum()) if total else 0
    false_count = total - true_count
    durations = (
        frame["duration_minutes"].to_numpy(dtype=float) if total else np.asarray([], dtype=float)
    )
    alarm_points = int(active.sum())
    total_points = int(mask.sum())
    summary = {
        "island_count": total,
        "true_positive_island_count": true_count,
        "false_alarm_island_count": false_count,
        "mean_duration_minutes": float(np.mean(durations)) if durations.size else None,
        "median_duration_minutes": float(np.median(durations)) if durations.size else None,
        "max_duration_minutes": float(np.max(durations)) if durations.size else None,
        "alarm_points": alarm_points,
        "total_evaluation_points": total_points,
        "coverage": alarm_points / total_points if total_points else 0.0,
        "coverage_percent": 100.0 * alarm_points / total_points if total_points else 0.0,
    }
    return frame, summary
