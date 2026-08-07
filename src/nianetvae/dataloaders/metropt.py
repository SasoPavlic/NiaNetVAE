"""Canonical MetroPT loading, feature engineering, and temporal study plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROLLING_AGGREGATIONS, DataConfig
from .preprocessing import FrozenPreprocessor
from .sequences import sequence_anchor_mask


@dataclass(frozen=True)
class MaintenanceEvent:
    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    severity: str


@dataclass(frozen=True)
class CyclePlan:
    cycle_id: int
    source_event_id: str
    update_start: pd.Timestamp | None
    update_end: pd.Timestamp | None
    score_start: pd.Timestamp
    score_end: pd.Timestamp
    update_possible: bool
    alias_to_cycle: int | None = None


@dataclass(frozen=True)
class PreparedMetroPTData:
    scaled_features: pd.DataFrame
    feature_names: tuple[str, ...]
    operation_phase: pd.Series
    events: tuple[MaintenanceEvent, ...]
    cycles: tuple[CyclePlan, ...]
    baseline_mask: pd.Series
    baseline_train_mask: pd.Series
    baseline_validation_mask: pd.Series
    calibration_mask: pd.Series
    post_maintenance_train_mask: pd.Series
    evaluation_mask: pd.Series
    preprocessor: FrozenPreprocessor
    dataset_hash: str
    feature_hash: str
    schedule_hash: str

    @property
    def data_contract_fingerprint(self) -> str:
        payload = {
            "dataset_hash": self.dataset_hash,
            "feature_hash": self.feature_hash,
            "schedule_hash": self.schedule_hash,
            "preprocessing_hash": self.preprocessor.fingerprint,
            "calibration_index_hash": _index_hash(
                self.calibration_mask[self.calibration_mask].index
            ),
            "evaluation_index_hash": _index_hash(self.evaluation_mask[self.evaluation_mask].index),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cycle_source_and_anchor_masks(
    prepared: PreparedMetroPTData,
    cycle: CyclePlan,
    test_phases: tuple[int, ...],
) -> tuple[pd.Series, pd.Series]:
    """Return the scoreable rows and frozen evaluation anchors for one cycle.

    Closely spaced maintenance events can legitimately produce a cycle with no
    evaluation anchors. Such cycles remain part of model lineage even though
    they contribute no predictions to the global evaluation population.
    """
    index = prepared.scaled_features.index
    start = cycle.score_start
    if cycle.cycle_id == 0:
        baseline_times = prepared.baseline_mask[prepared.baseline_mask].index
        start = pd.Timestamp(baseline_times.max())
    after_start = index > start
    before_end = (
        index <= cycle.score_end
        if cycle.cycle_id == len(prepared.cycles) - 1
        else index < cycle.score_end
    )
    source = pd.Series(after_start & before_end, index=index, dtype=bool)
    source &= prepared.operation_phase.isin(test_phases)
    source &= ~prepared.baseline_mask
    source &= ~prepared.post_maintenance_train_mask
    return source, prepared.evaluation_mask & source


def validate_cycle_evaluation_partition(
    prepared: PreparedMetroPTData,
    test_phases: tuple[int, ...],
) -> None:
    """Prove that cycle anchors partition the frozen evaluation population."""
    observed = pd.Series(False, index=prepared.evaluation_mask.index, dtype=bool)
    for cycle in prepared.cycles:
        _source, anchors = cycle_source_and_anchor_masks(prepared, cycle, test_phases)
        if (observed & anchors).any():
            raise ValueError(f"Cycle {cycle.cycle_id} overlaps an earlier evaluation population.")
        observed |= anchors
    if not observed.equals(prepared.evaluation_mask.astype(bool)):
        missing = int((prepared.evaluation_mask & ~observed).sum())
        unexpected = int((observed & ~prepared.evaluation_mask).sum())
        raise ValueError(
            "Cycle plan does not exactly partition the shared evaluation population; "
            f"missing={missing}, unexpected={unexpected}."
        )


def metropt_file_hash(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_hash(index: pd.Index) -> str:
    values = [pd.Timestamp(value).isoformat() for value in index]
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _infer_timestamp_column(frame: pd.DataFrame, configured: str | None) -> str:
    if configured:
        if configured not in frame.columns:
            raise ValueError(f"Configured timestamp column {configured!r} is missing.")
        return configured
    for candidate in ("timestamp", "Timestamp", "time", "Time", "datetime", "date", "Date"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not infer MetroPT timestamp column; configure data.timestamp_column.")


def load_metropt(path: str | Path, timestamp_column: str | None) -> tuple[pd.DataFrame, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MetroPT input not found: {source}")
    frame = pd.read_csv(source)
    unnamed = [column for column in frame.columns if str(column).lower().startswith("unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    timestamp = _infer_timestamp_column(frame, timestamp_column)
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    frame = frame.dropna(subset=[timestamp]).sort_values(timestamp).set_index(timestamp)
    if frame.index.has_duplicates:
        raise ValueError("MetroPT timestamps must be unique for a deterministic study timeline.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("MetroPT timestamps are not monotonically increasing.")
    return frame, metropt_file_hash(source)


def build_features(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    missing = [name for name in config.base_feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"MetroPT input is missing required features: {missing}")
    selected = frame.loc[:, list(config.base_feature_names)].apply(pd.to_numeric, errors="coerce")
    if selected.isna().all(axis=None):
        raise ValueError("MetroPT feature frame contains no numeric observations.")
    rolled = selected.rolling(config.rolling_window, min_periods=1).aggregate(ROLLING_AGGREGATIONS)
    rolled.columns = [f"{base}__{aggregation}" for base, aggregation in rolled.columns]
    rolled = rolled.ffill().bfill()
    if rolled.isna().any(axis=None):
        raise ValueError("Rolling feature construction produced unresolved missing values.")
    expected_count = len(config.base_feature_names) * len(ROLLING_AGGREGATIONS)
    if rolled.shape[1] != expected_count:
        raise ValueError(f"Expected {expected_count} rolling features, observed {rolled.shape[1]}.")
    return rolled.astype(np.float32)


def build_events(config: DataConfig) -> tuple[MaintenanceEvent, ...]:
    events = tuple(
        MaintenanceEvent(str(event_id), pd.Timestamp(start), pd.Timestamp(end), str(severity))
        for start, end, event_id, severity in config.maintenance_windows
    )
    if not events:
        raise ValueError("At least one maintenance event is required.")
    if tuple(sorted(events, key=lambda event: event.start)) != events:
        raise ValueError("Maintenance events must be sorted chronologically.")
    if any(event.end < event.start for event in events):
        raise ValueError("Maintenance event end precedes its start.")
    return events


def build_operation_phase(
    index: pd.DatetimeIndex,
    events: tuple[MaintenanceEvent, ...],
    pre_maintenance_minutes: int,
) -> pd.Series:
    values = np.zeros(len(index), dtype=np.int8)
    horizon = pd.Timedelta(minutes=int(pre_maintenance_minutes))
    for event in events:
        pre = (index >= event.start - horizon) & (index < event.start) & (values == 0)
        values[pre] = 1
        maintenance = (index >= event.start) & (index <= event.end)
        values[maintenance] = 2
    return pd.Series(values, index=index, name="operation_phase")


def _chronological_baseline_split(
    baseline_mask: pd.Series,
    validation_fraction: float,
    sequence_length: int,
) -> tuple[pd.Series, pd.Series]:
    positions = np.flatnonzero(baseline_mask.to_numpy(dtype=bool))
    if positions.size < 2 * sequence_length:
        raise ValueError(
            "Initial baseline is too short for non-overlapping train/validation sequences."
        )
    validation_rows = max(sequence_length, int(round(positions.size * validation_fraction)))
    validation_start_offset = positions.size - validation_rows
    embargo_rows = sequence_length - 1
    train_stop_offset = validation_start_offset - embargo_rows
    if train_stop_offset < sequence_length:
        raise ValueError("Initial baseline split leaves insufficient training rows.")
    train = pd.Series(False, index=baseline_mask.index, dtype=bool)
    validation = pd.Series(False, index=baseline_mask.index, dtype=bool)
    train.iloc[positions[:train_stop_offset]] = True
    validation.iloc[positions[validation_start_offset:]] = True
    return train, validation


def build_cycle_plan(
    index: pd.DatetimeIndex,
    events: tuple[MaintenanceEvent, ...],
    post_maintenance_train_minutes: int,
    sequence_length: int,
) -> tuple[tuple[CyclePlan, ...], pd.Series]:
    data_end = pd.Timestamp(index.max())
    cycles: list[CyclePlan] = []
    post_train = pd.Series(False, index=index, dtype=bool)
    cycles.append(
        CyclePlan(
            cycle_id=0,
            source_event_id="pre_W1",
            update_start=None,
            update_end=None,
            score_start=pd.Timestamp(index.min()),
            score_end=events[0].start,
            update_possible=True,
        )
    )
    latest_trainable = 0
    for offset, event in enumerate(events):
        next_start = events[offset + 1].start if offset + 1 < len(events) else data_end
        gap_start = event.end
        gap_end = next_start
        requested_cycle = offset + 1
        update_end = min(
            gap_start + pd.Timedelta(minutes=int(post_maintenance_train_minutes)),
            gap_end,
        )
        update_mask = (index > gap_start) & (index <= update_end)
        update_rows = int(update_mask.sum())
        update_possible = bool(gap_end > update_end and update_rows >= sequence_length)
        if update_possible:
            post_train.loc[update_mask] = True
            score_start = update_end
            alias_to = None
            latest_trainable = requested_cycle
        else:
            score_start = gap_start
            alias_to = latest_trainable
        cycles.append(
            CyclePlan(
                cycle_id=requested_cycle,
                source_event_id=event.event_id,
                update_start=gap_start if update_possible else None,
                update_end=update_end if update_possible else None,
                score_start=score_start,
                score_end=gap_end,
                update_possible=update_possible,
                alias_to_cycle=alias_to,
            )
        )
    return tuple(cycles), post_train


def _schedule_hash(events: tuple[MaintenanceEvent, ...], cycles: tuple[CyclePlan, ...]) -> str:
    payload = {
        "events": [
            {
                "id": event.event_id,
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
                "severity": event.severity,
            }
            for event in events
        ],
        "cycles": [
            {
                "cycle_id": cycle.cycle_id,
                "source_event_id": cycle.source_event_id,
                "update_start": cycle.update_start.isoformat()
                if cycle.update_start is not None
                else None,
                "update_end": cycle.update_end.isoformat()
                if cycle.update_end is not None
                else None,
                "score_start": cycle.score_start.isoformat(),
                "score_end": cycle.score_end.isoformat(),
                "update_possible": cycle.update_possible,
                "alias_to_cycle": cycle.alias_to_cycle,
            }
            for cycle in cycles
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare_metropt(config: DataConfig, preprocessing_policy: str) -> PreparedMetroPTData:
    source, dataset_hash = load_metropt(config.input_path, config.timestamp_column)
    raw_features = build_features(source, config)
    events = build_events(config)
    operation_phase = build_operation_phase(
        raw_features.index,
        events,
        config.pre_maintenance_minutes,
    )
    baseline_end = pd.Timestamp(raw_features.index.min()) + pd.Timedelta(
        minutes=int(config.initial_train_minutes)
    )
    eligible_train = operation_phase.isin(config.train_phases)
    baseline_mask = (
        pd.Series(
            (raw_features.index >= raw_features.index.min()) & (raw_features.index <= baseline_end),
            index=raw_features.index,
        )
        & eligible_train
    )
    baseline_train_mask, baseline_validation_mask = _chronological_baseline_split(
        baseline_mask,
        config.validation_fraction,
        config.sequence_length,
    )
    preprocessor = FrozenPreprocessor.fit(
        raw_features.loc[baseline_train_mask],
        binary_feature_names=config.binary_feature_names,
        policy=preprocessing_policy,
    )
    scaled_features = preprocessor.transform(raw_features)
    cycles, post_train_mask = build_cycle_plan(
        raw_features.index,
        events,
        config.post_maintenance_train_minutes,
        config.sequence_length,
    )
    calibration_mask = sequence_anchor_mask(
        baseline_mask,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    maintenance_mask = operation_phase == 2
    test_phase_mask = operation_phase.isin(config.test_phases)
    evaluation_rows = ~baseline_mask & ~post_train_mask & ~maintenance_mask & test_phase_mask
    # Every workflow is evaluated on the same end-anchor timestamps.  Row-based
    # methods intentionally use the recurrent models' sequence-anchor population
    # so coverage and event metrics cannot silently use different denominators.
    evaluation_mask = sequence_anchor_mask(
        evaluation_rows,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    feature_payload = json.dumps(list(raw_features.columns), separators=(",", ":"))
    feature_hash = hashlib.sha256(feature_payload.encode("utf-8")).hexdigest()
    prepared = PreparedMetroPTData(
        scaled_features=scaled_features,
        feature_names=tuple(str(column) for column in raw_features.columns),
        operation_phase=operation_phase,
        events=events,
        cycles=cycles,
        baseline_mask=baseline_mask,
        baseline_train_mask=baseline_train_mask,
        baseline_validation_mask=baseline_validation_mask,
        calibration_mask=calibration_mask,
        post_maintenance_train_mask=post_train_mask,
        evaluation_mask=evaluation_mask,
        preprocessor=preprocessor,
        dataset_hash=dataset_hash,
        feature_hash=feature_hash,
        schedule_hash=_schedule_hash(events, cycles),
    )
    validate_cycle_evaluation_partition(prepared, config.test_phases)
    return prepared
