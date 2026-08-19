"""Event-level early-warning metrics shared by all workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..dataloaders.sequences import true_runs


def normalize_windows(windows: Sequence) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    normalized: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in windows:
        start = pd.Timestamp(item.start if hasattr(item, "start") else item[0])
        end = pd.Timestamp(item.end if hasattr(item, "end") else item[1])
        if end >= start:
            normalized.append((start, end))
    return sorted(normalized)


def alarm_intervals(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if mask.empty:
        return []
    active = mask.fillna(False).astype(bool)
    return [
        (pd.Timestamp(active.index[positions[0]]), pd.Timestamp(active.index[positions[-1]]))
        for positions in true_runs(active)
    ]


def _mask_minutes(mask: pd.Series) -> float:
    return float(mask.fillna(False).astype(bool).sum()) * _sample_minutes(mask.index)


def _sample_minutes(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1.0
    deltas = np.diff(index.asi8) / 60_000_000_000.0
    positive = deltas[deltas > 0.0]
    return float(np.median(positive)) if positive.size else 1.0


def interval_duration_minutes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    index: pd.Index,
) -> float:
    return max(0.0, (end - start).total_seconds() / 60.0) + _sample_minutes(index)


def event_scores(
    predictions: pd.Series,
    windows: Sequence,
    early_warning_minutes: int,
) -> dict[str, Any]:
    events = normalize_windows(windows)
    intervals = alarm_intervals(predictions)
    used = [False] * len(intervals)
    horizon = pd.Timedelta(minutes=int(early_warning_minutes))
    tp = 0
    fn = 0
    for event_start, _event_end in events:
        matched = False
        for index, (alarm_start, alarm_end) in enumerate(intervals):
            if (
                not used[index]
                and alarm_start <= event_start
                and alarm_end >= event_start - horizon
            ):
                used[index] = True
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1
    fp = sum(not value for value in used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calculate_ttd(
    predictions: pd.Series, windows: Sequence, early_warning_minutes: int
) -> dict[str, Any]:
    values: list[float] = []
    missed = 0
    horizon = pd.Timedelta(minutes=int(early_warning_minutes))
    for event_start, _event_end in normalize_windows(windows):
        alarms = predictions.loc[event_start - horizon : event_start]
        alarms = alarms[alarms.astype(bool)]
        if alarms.empty:
            missed += 1
        else:
            values.append(float((event_start - alarms.index[0]).total_seconds() / 60.0))
    return {
        "ttd_values": values,
        "mean_ttd": float(np.mean(values)) if values else None,
        "std_ttd": float(np.std(values)) if values else None,
        "min_ttd": float(np.min(values)) if values else None,
        "max_ttd": float(np.max(values)) if values else None,
        "median_ttd": float(np.median(values)) if values else None,
        "detected_events": len(values),
        "missed_events": missed,
    }


def first_alarm_accuracy(
    predictions: pd.Series,
    windows: Sequence,
    early_warning_minutes: int,
) -> dict[str, Any]:
    intervals = alarm_intervals(predictions)
    tp_events = 0
    correct = 0
    horizon = pd.Timedelta(minutes=int(early_warning_minutes))
    for event_start, _event_end in normalize_windows(windows):
        warning_start = event_start - horizon
        starts = [
            start for start, end in intervals if start <= event_start and end >= warning_start
        ]
        if starts:
            tp_events += 1
            if warning_start <= min(starts) <= event_start:
                correct += 1
    return {
        "first_alarm_accuracy": correct / tp_events if tp_events else None,
        "tp_events": tp_events,
        "first_alarm_in_window": correct,
    }


def false_alarm_rate(
    predictions: pd.Series,
    windows: Sequence,
    early_warning_minutes: int,
    eval_mask: pd.Series,
) -> dict[str, Any]:
    intervals = alarm_intervals(predictions)
    horizon = pd.Timedelta(minutes=int(early_warning_minutes))
    events = normalize_windows(windows)
    false_count = sum(
        not any(start <= event_start and end >= event_start - horizon for event_start, _ in events)
        for start, end in intervals
    )
    total_days = _mask_minutes(eval_mask.astype(bool)) / 1440.0
    total_weeks = total_days / 7.0
    return {
        "total_alarm_intervals": len(intervals),
        "fp_intervals": int(false_count),
        "far_per_day": false_count / total_days if total_days else None,
        "far_per_week": false_count / total_weeks if total_weeks else None,
        "total_days": total_days,
        "total_weeks": total_weeks,
    }


def alarm_coverage(predictions: pd.Series, eval_mask: pd.Series) -> dict[str, Any]:
    mask = eval_mask.reindex(predictions.index).fillna(False).astype(bool)
    total = int(mask.sum())
    alarms = int((predictions.fillna(False).astype(bool) & mask).sum())
    coverage = alarms / total if total else 0.0
    return {
        "alarm_coverage": coverage,
        "alarm_coverage_percent": coverage * 100.0,
        "alarm_points": alarms,
        "total_points": total,
    }


def mtia(predictions: pd.Series) -> dict[str, Any]:
    durations = [
        interval_duration_minutes(start, end, predictions.index)
        for start, end in alarm_intervals(predictions)
    ]
    return {
        "mtia_minutes": float(np.mean(durations)) if durations else None,
        "std_minutes": float(np.std(durations)) if durations else None,
        "min_minutes": float(np.min(durations)) if durations else None,
        "max_minutes": float(np.max(durations)) if durations else None,
        "median_minutes": float(np.median(durations)) if durations else None,
        "num_intervals": len(durations),
        "durations": durations,
    }


def nab_score(
    predictions: pd.Series,
    windows: Sequence,
    early_warning_minutes: int,
    profile: str,
) -> dict[str, Any]:
    profiles = {
        "standard": {"tp": 1.0, "fp": 0.11, "fn": 1.0},
        "low_fp": {"tp": 1.0, "fp": 0.22, "fn": 1.0},
        "low_fn": {"tp": 1.0, "fp": 0.11, "fn": 2.0},
    }
    if profile not in profiles:
        raise ValueError(f"Unknown NAB profile={profile!r}.")
    weights = profiles[profile]
    intervals = alarm_intervals(predictions)
    used: set[int] = set()
    window_scores: list[float] = []
    for event_start, _event_end in normalize_windows(windows):
        warning_start = event_start - pd.Timedelta(minutes=int(early_warning_minutes))
        best = -weights["fn"]
        best_index = None
        for index, (start, end) in enumerate(intervals):
            if index in used or start > event_start or end < warning_start:
                continue
            detected = max(start, warning_start)
            position = (
                (detected - warning_start).total_seconds() / 60.0 / float(early_warning_minutes)
            )
            score = weights["tp"] / (1.0 + np.exp(5.0 * (position - 0.5)))
            if score > best:
                best = float(score)
                best_index = index
        if best_index is not None:
            used.add(best_index)
        window_scores.append(float(best))
    raw = float(sum(window_scores) - (len(intervals) - len(used)) * weights["fp"])
    maximum = len(window_scores) * weights["tp"]
    minimum = -len(window_scores) * weights["fn"] - len(intervals) * weights["fp"]
    normalized = 100.0 * (raw - minimum) / (maximum - minimum) if maximum != minimum else 0.0
    return {
        "nab_score_raw": raw,
        "nab_score_normalized": normalized,
        "profile": profile,
        "window_scores": window_scores,
        "num_fp": len(intervals) - len(used),
    }


def precision_recall_vs_leadtime(
    predictions: pd.Series,
    windows: Sequence,
    lead_times: Sequence[int],
    base_early_warning: int,
) -> dict[str, list]:
    events = normalize_windows(windows)
    intervals = alarm_intervals(predictions)
    output: dict[str, list] = {
        "lead_times": list(lead_times),
        "precision": [],
        "recall": [],
        "f1": [],
        "tp": [],
        "fp": [],
        "fn": [],
    }
    for lead in lead_times:
        used: set[int] = set()
        tp = 0
        for event_start, event_end in events:
            warning_start = event_start - pd.Timedelta(minutes=int(base_early_warning))
            deadline = event_start - pd.Timedelta(minutes=int(lead))
            for index, (start, end) in enumerate(intervals):
                if (
                    index not in used
                    and start <= event_end
                    and end >= warning_start
                    and start <= deadline
                ):
                    used.add(index)
                    tp += 1
                    break
        fn = len(events) - tp
        fp = len(intervals) - len(used)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        for key, value in (
            ("tp", tp),
            ("fp", fp),
            ("fn", fn),
            ("precision", precision),
            ("recall", recall),
            ("f1", f1),
        ):
            output[key].append(value)
    return output


def evaluate_maintenance_prediction(
    predictions: pd.Series,
    maintenance_windows: Sequence,
    early_warning_minutes: int,
    *,
    method_name: str,
    eval_mask: pd.Series,
    lead_step_minutes: int = 30,
    sensitivity_leads: Sequence[int] | None = None,
) -> dict[str, Any]:
    predictions = predictions.reindex(eval_mask.index).fillna(False).astype(bool)
    eval_mask = eval_mask.reindex(predictions.index).fillna(False).astype(bool)
    predictions &= eval_mask
    lead_times = (
        sorted({int(value) for value in sensitivity_leads if int(value) > 0})
        if sensitivity_leads is not None
        else list(range(lead_step_minutes, early_warning_minutes + 1, lead_step_minutes))
    )
    if not lead_times or lead_times[-1] != early_warning_minutes:
        lead_times.append(early_warning_minutes)
    ttd = calculate_ttd(predictions, maintenance_windows, early_warning_minutes)
    return {
        "method_name": method_name,
        "event_scores": event_scores(predictions, maintenance_windows, early_warning_minutes),
        "ttd": ttd,
        "lead_time_distribution": {
            "bins": list(range(0, early_warning_minutes + lead_step_minutes, lead_step_minutes)),
            "counts": np.histogram(
                ttd["ttd_values"],
                bins=list(range(0, early_warning_minutes + lead_step_minutes, lead_step_minutes)),
            )[0].tolist()
            if ttd["ttd_values"]
            else [0] * max(1, early_warning_minutes // lead_step_minutes),
        },
        "first_alarm_accuracy": first_alarm_accuracy(
            predictions, maintenance_windows, early_warning_minutes
        ),
        "far": false_alarm_rate(predictions, maintenance_windows, early_warning_minutes, eval_mask),
        "coverage": alarm_coverage(predictions, eval_mask),
        "mtia": mtia(predictions),
        "nab": {
            name: nab_score(predictions, maintenance_windows, early_warning_minutes, name)
            for name in ("standard", "low_fp", "low_fn")
        },
        "pr_leadtime": precision_recall_vs_leadtime(
            predictions, maintenance_windows, lead_times, early_warning_minutes
        ),
    }
