"""
MetroPT-3 DataLoader for NiaNetVAE.

This loader mirrors the MetroPT PdM framework's feature engineering and maintenance-cycle semantics:
  - Rolling aggregation (mean/median/std/skew/min/max) over a configurable time-based window.
  - Maintenance context via Davari et al. (2021) default failure windows and operation_phase labels:
      0 = normal, 1 = pre-maintenance, 2 = maintenance.

Two regimes are supported:
  - regime="single"
      Train: baseline = first train_minutes from dataset start (inclusive cutoff), phases in train_phases.
      Test:  (baseline_end, start_W1), phases in test_phases (default {0,1}).
  - regime="per_maint"
      cycle_id=0 maps to pre_W1 (baseline-trained model tested on baseline_end..W1_start).
      cycle_id=1..21 maps to Davari window order (#1..#21).
      post_train = [end_j, min(end_j + post_train_minutes, start_{j+1}))
      after_maint = [end(post_train), start_{j+1})   (or until end of data if j is the last window)
      Train: baseline ∪ post_train, phases in train_phases (typically {0,1}).
      Test:  after_maint, phases in test_phases (default {0,1}).

An explicit goal is to prevent sequence windows from crossing gaps between disjoint time blocks. To achieve this,
we build windows within each contiguous True-run of the selected masks (baseline/post-train/test) only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from log import Log
from nianetvae.dataloaders import BaseDataLoader


LIKELY_METROPT_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "Caudal_impulses",
]

ROLLING_AGGREGATIONS = ["mean", "median", "std", "skew", "min", "max"]

LEGACY_PREPROCESSING_POLICY = "standard_scaler_v1"
BINARY_AWARE_PREPROCESSING_POLICY = "binary_passthrough_v1"
SUPPORTED_PREPROCESSING_POLICIES = {
    LEGACY_PREPROCESSING_POLICY,
    BINARY_AWARE_PREPROCESSING_POLICY,
}
PREPROCESSING_POLICY_VERSIONS = {
    LEGACY_PREPROCESSING_POLICY: "1.0",
    BINARY_AWARE_PREPROCESSING_POLICY: "1.0",
}
LEGACY_VALIDATION_SPLIT_POLICY = "window_chronological_v1"
LEAKAGE_FREE_VALIDATION_SPLIT_POLICY = "raw_non_overlapping_v1"
SUPPORTED_VALIDATION_SPLIT_POLICIES = {
    LEGACY_VALIDATION_SPLIT_POLICY,
    LEAKAGE_FREE_VALIDATION_SPLIT_POLICY,
}

# These MetroPT channels are physical on/off controls or status indicators.
# The binary-aware policy leaves every rolling feature derived from them in its
# natural units instead of dividing rare state changes by a near-zero scale.
DEFAULT_METROPT_BINARY_FEATURES = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


# ===== MetroPT-3 failure windows (Davari et al., 2021) =====
# Table II intervals normalized to ISO (YYYY-MM-DD HH:MM:SS)
# Format: (start, end, id, severity)
DEFAULT_METROPT_WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2020-04-12 11:50:00", "2020-04-12 23:30:00", "#1", "high"),
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "#2", "high"),
    ("2020-04-19 00:00:00", "2020-04-19 01:30:00", "#3", "high"),
    ("2020-04-29 03:20:00", "2020-04-29 04:00:00", "#4", "high"),
    ("2020-04-29 22:00:00", "2020-04-29 22:20:00", "#5", "high"),
    ("2020-05-13 14:00:00", "2020-05-13 23:59:00", "#6", "high"),
    ("2020-05-18 05:00:00", "2020-05-18 05:30:00", "#7", "high"),
    ("2020-05-19 10:10:00", "2020-05-19 11:00:00", "#8", "high"),
    ("2020-05-19 22:10:00", "2020-05-19 23:59:00", "#9", "high"),
    ("2020-05-20 00:00:00", "2020-05-20 20:00:00", "#10", "high"),
    ("2020-05-23 09:50:00", "2020-05-23 10:10:00", "#11", "high"),
    ("2020-05-29 23:30:00", "2020-05-29 23:59:00", "#12", "high"),
    ("2020-05-30 00:00:00", "2020-05-30 06:00:00", "#13", "high"),
    ("2020-06-01 15:00:00", "2020-06-01 15:40:00", "#14", "high"),
    ("2020-06-03 10:00:00", "2020-06-03 11:00:00", "#15", "high"),
    ("2020-06-05 10:00:00", "2020-06-05 23:59:00", "#16", "high"),
    ("2020-06-06 00:00:00", "2020-06-06 23:59:00", "#17", "high"),
    ("2020-06-07 00:00:00", "2020-06-07 14:30:00", "#18", "high"),
    ("2020-07-08 17:30:00", "2020-07-08 19:00:00", "#19", "high"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "#20", "medium"),
    ("2020-07-17 04:30:00", "2020-07-17 05:30:00", "#21", "high"),
]


def convert_wsl_to_windows_path(path: str) -> str:
    """Convert /mnt/<drive>/... to Windows-style paths when running on Windows."""
    if platform.system() != "Windows":
        return path
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", path)
    if not m:
        return path
    drive = m.group(1).upper()
    rest = m.group(2) or ""
    return f"{drive}:{rest}".replace("/", "\\")


def infer_timestamp_column(df: pd.DataFrame, user_ts: Optional[str]) -> str:
    if user_ts and user_ts in df.columns:
        return user_ts
    for c in ["timestamp", "time", "datetime", "date", "Date", "Timestamp", "Time"]:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("Could not infer timestamp column. Provide data_params.timestamp_col.")


def load_csv(input_path: str, timestamp_col: Optional[str], drop_unnamed: bool) -> pd.DataFrame:
    """Load MetroPT CSV, parse timestamp column, and index by time."""
    input_path = convert_wsl_to_windows_path(str(input_path))
    df = pd.read_csv(input_path)
    if drop_unnamed:
        for c in list(df.columns):
            if str(c).lower().startswith("unnamed"):
                df = df.drop(columns=[c])
    ts_col = infer_timestamp_column(df, timestamp_col)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True).set_index(ts_col)
    return df


def select_numeric_features(df: pd.DataFrame, prefer: Optional[List[str]] = None) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if prefer:
        chosen = [c for c in prefer if c in num_cols]
        chosen += [c for c in num_cols if c not in chosen]
        return chosen
    return num_cols


def build_rolling_features(
    df_num: pd.DataFrame,
    rolling_window: str = "600s",
    min_periods: int = 1,
) -> pd.DataFrame:
    rolled = df_num.rolling(rolling_window, min_periods=min_periods)
    agg = rolled.aggregate(ROLLING_AGGREGATIONS)
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = ["__".join(map(str, col)).strip() for col in agg.columns.values]
    else:
        agg.columns = [str(col) for col in agg.columns]
    agg = agg.ffill().bfill()
    return agg


def build_operation_phase(
    index: pd.DatetimeIndex,
    windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp, Optional[str], Optional[str]]],
    pre_minutes: float = 120.0,
) -> pd.Series:
    """
    Build an operation phase indicator:
      0 = normal, 1 = pre-maintenance, 2 = maintenance.
    Maintenance overrides pre-maintenance when overlapping.
    """
    phase = pd.Series(np.zeros(len(index), dtype=np.int8), index=index, name="operation_phase")
    if index.size == 0 or not windows:
        return phase
    try:
        pre_delta = pd.to_timedelta(float(pre_minutes), unit="m")
    except Exception:
        pre_delta = pd.to_timedelta(0, unit="h")

    arr = phase.to_numpy()
    for item in windows:
        if len(item) < 2:
            continue
        start = pd.to_datetime(item[0])
        end = pd.to_datetime(item[1])
        if pd.isna(start) or pd.isna(end) or end < start:
            continue

        maint_mask = (index >= start) & (index <= end)
        if maint_mask.any():
            arr[maint_mask] = np.int8(2)

        if pre_delta is not None and pre_delta > pd.Timedelta(0):
            pre_start = start - pre_delta
            pre_mask = (index >= pre_start) & (index < start)
            if pre_mask.any():
                zero_mask = arr == 0
                combined = pre_mask & zero_mask
                if combined.any():
                    arr[combined] = np.int8(1)

    phase[:] = arr
    return phase


def _as_int_list(values: Optional[Iterable[object]], default: Sequence[int]) -> List[int]:
    if values is None:
        return list(default)
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    if isinstance(values, str):
        parts = [p.strip() for p in values.split(",") if p.strip()]
        return [int(p) for p in parts]
    return [int(v) for v in list(values)]


def build_feature_hash(feature_names: Sequence[str]) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_metropt_preprocessing_policy(value: Optional[str]) -> str:
    """Resolve the versioned MetroPT preprocessing policy."""
    policy = str(value or LEGACY_PREPROCESSING_POLICY).strip().lower()
    if policy not in SUPPORTED_PREPROCESSING_POLICIES:
        allowed = ", ".join(sorted(SUPPORTED_PREPROCESSING_POLICIES))
        raise ValueError(
            f"Unsupported MetroPT preprocessing_policy={value!r}. Allowed: {allowed}."
        )
    return policy


def resolve_validation_split_policy(value: Optional[str]) -> str:
    """Resolve the versioned train/validation separation policy."""
    policy = str(value or LEGACY_VALIDATION_SPLIT_POLICY).strip().lower()
    if policy not in SUPPORTED_VALIDATION_SPLIT_POLICIES:
        allowed = ", ".join(sorted(SUPPORTED_VALIDATION_SPLIT_POLICIES))
        raise ValueError(
            f"Unsupported MetroPT validation_split_policy={value!r}. Allowed: {allowed}."
        )
    return policy


def _as_feature_name_list(
    values: Optional[Sequence[str]],
    default: Sequence[str],
) -> List[str]:
    if values is None:
        return [str(value) for value in default]
    if isinstance(values, str):
        return [part.strip() for part in values.split(",") if part.strip()]
    return [str(value).strip() for value in values if str(value).strip()]


def fit_metropt_preprocessing_scaler(
    train_segments: Sequence[np.ndarray],
    feature_names: Sequence[str],
    *,
    policy: Optional[str] = None,
    binary_feature_names: Optional[Sequence[str]] = None,
) -> Tuple[StandardScaler, Dict[str, object]]:
    """Fit the selected scaler without changing feature order or dimensionality.

    ``standard_scaler_v1`` reproduces the historical behavior exactly.
    ``binary_passthrough_v1`` still fits one ordinary sklearn StandardScaler,
    but sets mean=0 and scale=1 for rolling columns derived from declared
    binary controls. The serialized scaler therefore remains consumable by
    existing inference code that only calls ``transform``.
    """
    resolved_policy = resolve_metropt_preprocessing_policy(policy)
    ordered_feature_names = [str(name) for name in feature_names]
    if not ordered_feature_names:
        raise ValueError("MetroPT preprocessing requires at least one feature name.")

    scaler = StandardScaler()
    fitted_rows = 0
    for segment in train_segments:
        values = np.asarray(segment)
        if values.ndim != 2 or values.shape[1] != len(ordered_feature_names):
            raise ValueError(
                "MetroPT preprocessing segment shape does not match feature contract: "
                f"shape={values.shape} features={len(ordered_feature_names)}."
            )
        if values.shape[0] > 0:
            scaler.partial_fit(values)
            fitted_rows += int(values.shape[0])
    if fitted_rows < 1:
        raise ValueError("MetroPT preprocessing cannot fit a scaler on zero rows.")

    configured_binary_features = _as_feature_name_list(
        binary_feature_names,
        default=DEFAULT_METROPT_BINARY_FEATURES,
    )
    matched_binary_features: List[str] = []
    binary_derived_indices: List[int] = []
    for binary_name in configured_binary_features:
        prefix = f"{binary_name}__"
        matches = [
            index
            for index, feature_name in enumerate(ordered_feature_names)
            if feature_name.startswith(prefix)
        ]
        if matches:
            matched_binary_features.append(binary_name)
            binary_derived_indices.extend(matches)
    binary_derived_indices = sorted(set(binary_derived_indices))
    binary_derived_feature_names = [
        ordered_feature_names[index] for index in binary_derived_indices
    ]

    passthrough_indices: List[int] = []
    passthrough_feature_names: List[str] = []
    applied_binary_features: List[str] = []

    if resolved_policy == BINARY_AWARE_PREPROCESSING_POLICY:
        applied_binary_features = list(matched_binary_features)
        passthrough_indices = list(binary_derived_indices)
        if not passthrough_indices:
            raise ValueError(
                "binary_passthrough_v1 matched no engineered features. "
                "Check data_params.binary_feature_names against the MetroPT columns."
            )
        passthrough_feature_names = [
            ordered_feature_names[index] for index in passthrough_indices
        ]
        index_array = np.asarray(passthrough_indices, dtype=np.int64)
        scaler.mean_[index_array] = 0.0
        scaler.scale_[index_array] = 1.0
        scaler.var_[index_array] = 1.0

    # These extra attributes are safe for sklearn/joblib consumers and make the
    # transformation self-describing even before model metadata is inspected.
    scaler.nianetvae_preprocessing_policy_ = resolved_policy
    scaler.nianetvae_preprocessing_policy_version_ = PREPROCESSING_POLICY_VERSIONS[
        resolved_policy
    ]
    scaler.nianetvae_passthrough_indices_ = np.asarray(
        passthrough_indices, dtype=np.int64
    )

    passthrough_index_set = set(passthrough_indices)
    standardized_indices = [
        index
        for index in range(len(ordered_feature_names))
        if index not in passthrough_index_set
    ]
    report: Dict[str, object] = {
        "policy": resolved_policy,
        "policy_version": PREPROCESSING_POLICY_VERSIONS[resolved_policy],
        "behavior": (
            "all_engineered_features_standardized"
            if resolved_policy == LEGACY_PREPROCESSING_POLICY
            else "continuous_derived_standardized_binary_derived_passthrough"
        ),
        "preserves_feature_order": True,
        "preserves_feature_count": True,
        "configured_binary_feature_names": configured_binary_features,
        "matched_binary_feature_names": matched_binary_features,
        "binary_derived_feature_indices": binary_derived_indices,
        "binary_derived_feature_names": binary_derived_feature_names,
        "binary_derived_feature_count": len(binary_derived_indices),
        "applied_binary_feature_names": applied_binary_features,
        "passthrough_feature_indices": passthrough_indices,
        "passthrough_feature_names": passthrough_feature_names,
        "passthrough_feature_count": len(passthrough_indices),
        "standardized_feature_indices": standardized_indices,
        "standardized_feature_count": len(standardized_indices),
        "fitted_row_count": fitted_rows,
    }
    return scaler, report


def _segments_from_mask(values: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Extract contiguous segments (slices) from values where mask is True."""
    if values.shape[0] != mask.shape[0]:
        raise ValueError("Mask length does not match data length.")
    segments: List[np.ndarray] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append(values[start:i])
            start = None
    if start is not None:
        segments.append(values[start:])
    return segments


def _segment_metadata_from_mask(
    index: pd.DatetimeIndex,
    mask: np.ndarray,
    op_phase: Optional[pd.Series] = None,
) -> List[Dict[str, object]]:
    """Describe contiguous True-runs in a mask without exposing data values."""
    if len(index) != mask.shape[0]:
        raise ValueError("Mask length does not match index length.")
    phase_vals = None
    if op_phase is not None:
        phase_vals = op_phase.reindex(index).to_numpy(dtype=np.int8, copy=False)

    segments: List[Dict[str, object]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append(_build_segment_metadata(index, phase_vals, start, i))
            start = None
    if start is not None:
        segments.append(_build_segment_metadata(index, phase_vals, start, len(index)))
    return segments


def _build_segment_metadata(
    index: pd.DatetimeIndex,
    phase_vals: Optional[np.ndarray],
    start: int,
    stop: int,
) -> Dict[str, object]:
    item: Dict[str, object] = {
        "start": pd.to_datetime(index[start]).isoformat(),
        "end": pd.to_datetime(index[stop - 1]).isoformat(),
        "start_pos": int(start),
        "end_pos": int(stop - 1),
        "rows": int(stop - start),
    }
    if phase_vals is not None:
        phase_slice = phase_vals[start:stop]
        item["phase_counts"] = {
            str(phase): int(np.sum(phase_slice == phase))
            for phase in sorted(set(int(v) for v in phase_slice.tolist()))
        }
    return item


class MetroPTSegmentedSequenceDataset(Dataset):
    """Sliding-window dataset over multiple contiguous segments (no cross-gap windows)."""

    def __init__(
        self,
        segments: List[np.ndarray],
        phase_segments: Optional[List[np.ndarray]] = None,
        seq_len: int = 200,
        stride: int = 1,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        self._segments: List[np.ndarray] = []
        self._phase_segments: Optional[List[np.ndarray]] = [] if phase_segments is not None else None
        self._windows_per_segment: List[int] = []
        self.window_positive_count = 0
        self.window_negative_count = 0

        if phase_segments is not None and len(phase_segments) != len(segments or []):
            raise ValueError("phase_segments length must match segments length.")

        for idx, seg in enumerate(segments or []):
            arr = np.asarray(seg, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValueError("Each segment must be a 1D or 2D array.")
            self._segments.append(arr)
            n = arr.shape[0]
            w = max(0, (n - self.seq_len) // self.stride + 1)
            self._windows_per_segment.append(int(w))
            if self._phase_segments is not None:
                phase_arr = np.asarray(phase_segments[idx], dtype=np.int8).reshape(-1)
                if phase_arr.shape[0] != n:
                    raise ValueError("Each phase segment must have the same row count as its signal segment.")
                self._phase_segments.append(phase_arr)
                if w > 0:
                    anchors = np.arange(self.seq_len - 1, self.seq_len - 1 + w * self.stride, self.stride, dtype=np.int64)
                    positives = int(np.sum(phase_arr[anchors] == 1))
                    self.window_positive_count += positives
                    self.window_negative_count += int(w - positives)

        self._cum_windows = np.cumsum(self._windows_per_segment, dtype=np.int64)
        self._total_windows = int(self._cum_windows[-1]) if self._cum_windows.size else 0

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx < 0 or idx >= self._total_windows:
            raise IndexError("Index out of range in MetroPTSegmentedSequenceDataset.")
        seg_idx = int(np.searchsorted(self._cum_windows, idx, side="right"))
        prev = int(self._cum_windows[seg_idx - 1]) if seg_idx > 0 else 0
        local = int(idx - prev)
        start = local * self.stride
        window = self._segments[seg_idx][start : start + self.seq_len]
        signal = torch.from_numpy(window).float()
        operation_phase = 0
        target = 0
        if self._phase_segments is not None:
            anchor = start + self.seq_len - 1
            operation_phase = int(self._phase_segments[seg_idx][anchor])
            target = 1 if operation_phase == 1 else 0
        return {"signal": signal, "target": target, "operation_phase": operation_phase, "ts_id": seg_idx}


class MetroPTDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        batch_size: int,
        seq_len: int,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        val_size: float,
        data_percentage: float,
        rolling_window: str = "60s",
        train_minutes: float = 1440.0,
        post_train_minutes: float = 1440.0,
        pre_maint_minutes: float = 120.0,
        regime: str = "single",
        cycle_id: int = 1,
        stride: int = 10,
        timestamp_col: Optional[str] = None,
        drop_unnamed_index: bool = True,
        train_phases: Optional[Sequence[int]] = (0, 1),
        test_phases: Optional[Sequence[int]] = (0, 1),
        preprocessing_policy: str = LEGACY_PREPROCESSING_POLICY,
        binary_feature_names: Optional[Sequence[str]] = None,
        validation_split_policy: str = LEGACY_VALIDATION_SPLIT_POLICY,
        shuffle_train: bool = False,
        drop_last_train: bool = True,
        train_shuffle_seed: Optional[int] = None,
        workflow_mode: Optional[str] = None,
        finetune_data_policy: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            seq_len=seq_len,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            val_size=val_size,
            data_percentage=data_percentage,
            **kwargs,
        )

        self.rolling_window = str(rolling_window)
        self.train_minutes = float(train_minutes)
        self.post_train_minutes = float(post_train_minutes)
        self.pre_maint_minutes = float(pre_maint_minutes)
        self.regime = str(regime).strip().lower()
        self.cycle_id = int(cycle_id)
        self.stride = int(stride)
        self.timestamp_col = timestamp_col
        self.drop_unnamed_index = bool(drop_unnamed_index)
        self.train_phases = _as_int_list(train_phases, default=(0, 1))
        self.test_phases = _as_int_list(test_phases, default=(0, 1))
        self.preprocessing_policy = resolve_metropt_preprocessing_policy(
            preprocessing_policy
        )
        self.binary_feature_names = _as_feature_name_list(
            binary_feature_names,
            default=DEFAULT_METROPT_BINARY_FEATURES,
        )
        self.validation_split_policy = resolve_validation_split_policy(
            validation_split_policy
        )
        self.workflow_mode = str(workflow_mode or "").strip().lower()
        self.finetune_data_policy = dict(finetune_data_policy or {})
        self._finetune_data_policy_active = bool(
            self.finetune_data_policy.get("enabled", False)
            and self.workflow_mode == "per_maint_finetune_search"
            and self.regime == "per_maint"
            and self.cycle_id > 0
        )
        self._train_shuffle = bool(shuffle_train)
        self._train_drop_last = bool(drop_last_train)
        self._train_shuffle_seed = int(
            train_shuffle_seed
            if train_shuffle_seed is not None
            else self.finetune_data_policy.get("random_seed", 42)
        )

        self.n_features: Optional[int] = None
        self.base_feature_names: List[str] = []
        self.rolling_feature_names: List[str] = []
        self.rolling_aggregations: List[str] = list(ROLLING_AGGREGATIONS)
        self.feature_hash: Optional[str] = None
        self.scaler: Optional[StandardScaler] = None
        self.preprocessing_report: Dict[str, object] = {}
        self.validation_split_report: Dict[str, object] = {}
        self._finetune_split_plan: Optional[Dict[str, object]] = None
        self.train_segment_metadata: List[Dict[str, object]] = []
        self.test_segment_metadata: List[Dict[str, object]] = []
        self.split_info: Dict[str, object] = {}
        self._summary_logged = False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _validate_finetune_data_policy(self) -> None:
        if not self._finetune_data_policy_active:
            return

        baseline_fraction = float(self.finetune_data_policy.get("baseline_replay_fraction", 0.5))
        local_val_fraction = float(self.finetune_data_policy.get("local_validation_fraction", 0.2))
        if not 0.0 <= baseline_fraction < 1.0:
            raise ValueError(
                "workflow.finetune.data_policy.baseline_replay_fraction must be in [0, 1)."
            )
        if not 0.0 < local_val_fraction < 1.0:
            raise ValueError(
                "workflow.finetune.data_policy.local_validation_fraction must be in (0, 1)."
            )
        short_local_fallback = str(
            self.finetune_data_policy.get(
                "short_local_fallback", "train_all_fixed_min_epochs"
            )
        ).strip().lower()
        if short_local_fallback not in {"train_all_fixed_min_epochs", "error"}:
            raise ValueError(
                "workflow.finetune.data_policy.short_local_fallback must be "
                "'train_all_fixed_min_epochs' or 'error'."
            )

    @staticmethod
    def _evenly_spaced_indices(total: int, count: int) -> List[int]:
        total = int(total)
        count = int(count)
        if total <= 0 or count <= 0:
            return []
        if count >= total:
            return list(range(total))
        positions = np.floor((np.arange(count, dtype=np.float64) + 0.5) * total / count).astype(np.int64)
        return positions.tolist()

    def _sequence_window_count(self, row_count: int) -> int:
        row_count = int(row_count)
        if row_count < self.seq_len:
            return 0
        return 1 + (row_count - self.seq_len) // self.stride

    def _overlap_embargo_window_count(self) -> int:
        """Return the minimum skipped starts needed for disjoint raw windows."""
        return int(max(0, self.seq_len - 1) // self.stride)

    def _resolve_finetune_split_plan(self, local_row_count: int) -> Dict[str, object]:
        """Plan the local chronological split once for scaling and datasets."""
        if self._finetune_split_plan is not None:
            return dict(self._finetune_split_plan)

        local_total = self._sequence_window_count(local_row_count)
        if local_total < 1:
            raise ValueError("Fine-tune data policy produced zero local fine-tune windows.")
        local_val_fraction = float(
            self.finetune_data_policy.get("local_validation_fraction", 0.2)
        )
        local_val_windows = max(1, int(np.floor(local_total * local_val_fraction)))
        local_val_start = local_total - local_val_windows
        embargo_enabled = bool(
            self.finetune_data_policy.get("validation_embargo", True)
        )
        requested_embargo_windows = (
            self._overlap_embargo_window_count()
            if embargo_enabled
            else 0
        )
        local_train_stop = local_val_start - requested_embargo_windows
        short_local_fallback = str(
            self.finetune_data_policy.get(
                "short_local_fallback", "train_all_fixed_min_epochs"
            )
        ).strip().lower()
        fallback_applied = bool(local_train_stop < 1)
        fallback_reason = None
        if fallback_applied:
            fallback_reason = (
                "insufficient_local_windows_for_non_overlapping_train_validation_split:"
                f"local={local_total},validation={local_val_windows},"
                f"requested_embargo={requested_embargo_windows}"
            )
            if short_local_fallback == "error":
                raise ValueError(
                    "Fine-tune local segment is too short for chronological "
                    "train/validation splitting with embargo: "
                    f"local_windows={local_total}, validation_windows={local_val_windows}, "
                    f"embargo_windows={requested_embargo_windows}."
                )
            local_train_indices = list(range(local_total))
            local_val_indices: List[int] = []
            applied_embargo_windows = 0
            validation_strategy = "disabled_short_local_fallback"
            early_stopping_eligible = False
            validation_start_row = int(local_row_count)
            scaler_local_train_rows = int(local_row_count)
        else:
            local_train_indices = list(range(local_train_stop))
            local_val_indices = list(range(local_val_start, local_total))
            applied_embargo_windows = requested_embargo_windows
            validation_strategy = "chronological_non_overlapping_local"
            early_stopping_eligible = True
            validation_start_row = int(local_val_start * self.stride)
            scaler_local_train_rows = int(
                (local_train_stop - 1) * self.stride + self.seq_len
            )
            if scaler_local_train_rows > validation_start_row:
                raise RuntimeError(
                    "Fine-tune split planner produced overlapping raw train/validation rows."
                )

        plan: Dict[str, object] = {
            "local_total_windows": int(local_total),
            "local_train_indices": local_train_indices,
            "local_validation_indices": local_val_indices,
            "requested_local_validation_windows": int(local_val_windows),
            "requested_embargo_windows": int(requested_embargo_windows),
            "applied_embargo_windows": int(applied_embargo_windows),
            "short_local_fallback": short_local_fallback,
            "short_local_fallback_applied": fallback_applied,
            "short_local_fallback_reason": fallback_reason,
            "validation_strategy": validation_strategy,
            "early_stopping_eligible": early_stopping_eligible,
            "validation_start_row": validation_start_row,
            "scaler_local_train_rows": scaler_local_train_rows,
            "unused_boundary_raw_rows": int(
                max(0, validation_start_row - scaler_local_train_rows)
            ),
            "validation_rows_excluded_from_scaler": int(
                max(0, int(local_row_count) - validation_start_row)
            ),
            "total_local_rows_excluded_from_scaler": int(
                max(0, int(local_row_count) - scaler_local_train_rows)
            ),
        }
        self._finetune_split_plan = dict(plan)
        return plan

    def _split_raw_validation_segment(
        self,
        train_segments_raw: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, object]]:
        """Split the chronological validation tail before fitting preprocessing.

        Validation windows are counted across the flattened, chronological
        segment order.  The split can therefore start in an earlier segment
        when the final post-maintenance segment is smaller than the requested
        validation fraction.  At most one segment is split; later segments are
        assigned wholly to validation and are excluded from scaler fitting.
        """
        if not train_segments_raw:
            raise ValueError("Cannot split validation from an empty training segment list.")
        window_counts = [self._sequence_window_count(len(segment)) for segment in train_segments_raw]
        total_windows = int(sum(window_counts))
        if total_windows < 2:
            raise ValueError("Not enough training windows for chronological validation.")
        val_windows = max(1, int(np.floor(total_windows * (float(self.val_size) / 100.0))))
        if val_windows >= total_windows:
            val_windows = total_windows - 1

        requested_embargo_windows = self._overlap_embargo_window_count()
        remaining_validation_windows = int(val_windows)
        split_segment_index: Optional[int] = None
        split_segment_validation_windows = 0
        for segment_index in range(len(train_segments_raw) - 1, -1, -1):
            segment_windows = int(window_counts[segment_index])
            if segment_windows < 1:
                continue
            if remaining_validation_windows <= segment_windows:
                split_segment_index = segment_index
                split_segment_validation_windows = remaining_validation_windows
                break
            remaining_validation_windows -= segment_windows

        if split_segment_index is None:
            raise RuntimeError("Could not locate the raw validation split boundary.")

        split_segment = train_segments_raw[split_segment_index]
        split_segment_windows = int(window_counts[split_segment_index])
        validation_start_window = int(
            split_segment_windows - split_segment_validation_windows
        )
        train_segments = list(train_segments_raw[:split_segment_index])
        validation_segments: List[np.ndarray]
        applied_embargo_windows = 0
        unused_boundary_raw_rows = 0
        train_raw_end: Optional[int] = None
        validation_raw_start = 0

        if validation_start_window == 0:
            # The requested tail begins exactly at this segment boundary.
            validation_segments = list(train_segments_raw[split_segment_index:])
        else:
            train_stop_window = (
                validation_start_window - requested_embargo_windows
            )
            if train_stop_window < 1:
                # There is no non-overlapping training window to preserve in
                # this segment. Assign it wholly to validation; the overshoot
                # is bounded by the overlap embargo and keeps earlier segments
                # available for training.
                validation_start_window = 0
                validation_segments = list(
                    train_segments_raw[split_segment_index:]
                )
            else:
                applied_embargo_windows = requested_embargo_windows
                train_raw_end = int(
                    (train_stop_window - 1) * self.stride + self.seq_len
                )
                validation_raw_start = int(validation_start_window * self.stride)
                if train_raw_end > validation_raw_start:
                    raise RuntimeError(
                        "Raw validation split produced overlapping train/validation observations."
                    )
                train_segments.append(split_segment[:train_raw_end])
                validation_segments = [
                    split_segment[validation_raw_start:]
                ] + list(train_segments_raw[split_segment_index + 1 :])
                unused_boundary_raw_rows = int(
                    validation_raw_start - train_raw_end
                )

        train_windows_actual = int(
            sum(self._sequence_window_count(len(segment)) for segment in train_segments)
        )
        val_windows_actual = int(
            sum(
                self._sequence_window_count(len(segment))
                for segment in validation_segments
            )
        )
        if train_windows_actual < 1 or val_windows_actual < 1:
            raise ValueError(
                "Leakage-free chronological split produced an empty train or validation dataset."
            )
        validation_rows = int(sum(len(segment) for segment in validation_segments))
        original_rows = int(sum(len(segment) for segment in train_segments_raw))
        scaler_rows = int(sum(len(segment) for segment in train_segments))
        report: Dict[str, object] = {
            "policy": LEAKAGE_FREE_VALIDATION_SPLIT_POLICY,
            "validation_strategy": "raw_chronological_non_overlapping_tail",
            "train_windows": train_windows_actual,
            "validation_windows": int(val_windows_actual),
            "requested_validation_windows": int(val_windows),
            "requested_sequence_embargo_windows": requested_embargo_windows,
            "sequence_embargo_windows": applied_embargo_windows,
            "split_segment_index": int(split_segment_index),
            "train_raw_end": train_raw_end,
            "validation_raw_start": validation_raw_start,
            "unused_boundary_raw_rows": unused_boundary_raw_rows,
            "validation_rows_excluded_from_scaler": validation_rows,
            "total_rows_excluded_from_scaler": int(original_rows - scaler_rows),
            "early_stopping_eligible": True,
        }
        return train_segments, validation_segments, report

    def _build_finetune_datasets(self, train_segments: List[np.ndarray]) -> None:
        """Build local-first fine-tune datasets with deterministic baseline replay."""
        self._validate_finetune_data_policy()
        if len(train_segments) < 2:
            raise ValueError(
                "Fine-tune data policy produced zero local fine-tune segments after phase filtering."
            )

        baseline_dataset = MetroPTSegmentedSequenceDataset(
            train_segments[:-1], seq_len=self.seq_len, stride=self.stride
        )
        local_dataset = MetroPTSegmentedSequenceDataset(
            [train_segments[-1]], seq_len=self.seq_len, stride=self.stride
        )
        baseline_total = int(len(baseline_dataset))
        local_total = int(len(local_dataset))
        if baseline_total < 1:
            raise ValueError("Fine-tune data policy produced zero baseline replay windows.")
        if local_total < 1:
            raise ValueError("Fine-tune data policy produced zero local fine-tune windows.")

        split_plan = self._resolve_finetune_split_plan(len(train_segments[-1]))
        local_val_fraction = float(self.finetune_data_policy.get("local_validation_fraction", 0.2))
        local_val_windows = int(split_plan["requested_local_validation_windows"])
        requested_embargo_windows = int(split_plan["requested_embargo_windows"])
        local_train_indices = list(split_plan["local_train_indices"])
        local_val_indices = list(split_plan["local_validation_indices"])
        applied_embargo_windows = int(split_plan["applied_embargo_windows"])
        short_local_fallback = str(split_plan["short_local_fallback"])
        fallback_applied = bool(split_plan["short_local_fallback_applied"])
        fallback_reason = split_plan["short_local_fallback_reason"]
        validation_strategy = str(split_plan["validation_strategy"])
        early_stopping_eligible = bool(split_plan["early_stopping_eligible"])

        baseline_fraction = float(self.finetune_data_policy.get("baseline_replay_fraction", 0.5))
        if baseline_fraction <= 0.0:
            baseline_replay_windows = 0
        else:
            baseline_replay_windows = int(
                round(len(local_train_indices) * baseline_fraction / (1.0 - baseline_fraction))
            )
            baseline_replay_windows = max(1, min(baseline_total, baseline_replay_windows))
        baseline_indices = self._evenly_spaced_indices(baseline_total, baseline_replay_windows)

        train_parts = []
        if baseline_indices:
            train_parts.append(Subset(baseline_dataset, baseline_indices))
        train_parts.append(Subset(local_dataset, local_train_indices))
        self.train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        self.val_dataset = Subset(local_dataset, local_val_indices) if local_val_indices else None
        self._train_shuffle = bool(self.finetune_data_policy.get("shuffle_train", True))
        self.split_info["shuffle_train"] = self._train_shuffle
        self.split_info["drop_last_train"] = self._train_drop_last
        self.split_info["train_shuffle_seed"] = self._train_shuffle_seed

        train_total = int(len(self.train_dataset))
        effective_baseline_fraction = (
            float(len(baseline_indices)) / float(train_total) if train_total > 0 else 0.0
        )
        policy_report = {
            "enabled": True,
            "mode": "local_train_with_baseline_replay",
            "baseline_total_windows": baseline_total,
            "baseline_replay_windows": int(len(baseline_indices)),
            "local_total_windows": local_total,
            "local_train_windows": int(len(local_train_indices)),
            "requested_local_validation_windows": int(local_val_windows),
            "local_validation_windows": int(len(local_val_indices)),
            "requested_local_validation_embargo_windows": int(requested_embargo_windows),
            "local_validation_embargo_windows": int(applied_embargo_windows),
            "unused_local_embargo_windows": int(applied_embargo_windows),
            "validation_strategy": validation_strategy,
            "early_stopping_eligible": bool(early_stopping_eligible),
            "short_local_fallback": short_local_fallback,
            "short_local_fallback_applied": bool(fallback_applied),
            "short_local_fallback_reason": fallback_reason,
            "requested_baseline_replay_fraction": baseline_fraction,
            "effective_baseline_replay_fraction": effective_baseline_fraction,
            "local_validation_fraction": local_val_fraction,
            "effective_local_validation_fraction": (
                float(len(local_val_indices)) / float(local_total)
                if local_total > 0
                else 0.0
            ),
            "shuffle_train": self._train_shuffle,
            "random_seed": self._train_shuffle_seed,
            "validation_rows_excluded_from_scaler": int(
                split_plan["validation_rows_excluded_from_scaler"]
            ),
            "unused_boundary_raw_rows": int(
                split_plan["unused_boundary_raw_rows"]
            ),
            "total_local_rows_excluded_from_scaler": int(
                split_plan["total_local_rows_excluded_from_scaler"]
            ),
        }
        self.split_info["fine_tune_data_policy"] = policy_report
        Log.info(
            "FINETUNE_DATA_POLICY "
            f"cycle_id={self.cycle_id:02d} baseline_total={baseline_total} "
            f"baseline_replay={len(baseline_indices)} local_total={local_total} "
            f"local_train={len(local_train_indices)} local_val={len(local_val_indices)} "
            f"local_embargo={applied_embargo_windows} "
            f"validation_strategy={validation_strategy} "
            f"early_stopping_eligible={str(early_stopping_eligible).lower()} "
            f"effective_baseline_fraction={effective_baseline_fraction:.4f} "
            f"shuffle_train={str(self._train_shuffle).lower()}"
        )

    def _default_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, str]]:
        out: List[Tuple[pd.Timestamp, pd.Timestamp, str, str]] = []
        for s, e, wid, sev in DEFAULT_METROPT_WINDOWS:
            out.append((pd.to_datetime(s), pd.to_datetime(e), wid, sev))
        out.sort(key=lambda t: t[0])
        return out

    def _baseline_bounds(self, index: pd.DatetimeIndex) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if index.empty:
            raise ValueError("Empty index in MetroPT data.")
        start = pd.to_datetime(index.min())
        end = start + pd.Timedelta(minutes=float(self.train_minutes))
        return start, end

    def _build_masks(
        self,
        index: pd.DatetimeIndex,
        op_phase: pd.Series,
        windows: List[Tuple[pd.Timestamp, pd.Timestamp, str, str]],
    ) -> Tuple[pd.Series, pd.Series]:
        train_phases = set(int(p) for p in self.train_phases)
        test_phases = set(int(p) for p in self.test_phases)

        # Phase 2 must be excluded everywhere in this adaptation.
        if 2 in train_phases:
            train_phases.remove(2)
        if 2 in test_phases:
            test_phases.remove(2)

        base_start, base_end = self._baseline_bounds(index)
        baseline_mask = (index >= base_start) & (index <= base_end)

        if not windows:
            raise ValueError("No maintenance windows are configured for MetroPT.")

        w1_start = pd.to_datetime(windows[0][0])

        info: Dict[str, object] = {
            "regime": self.regime,
            "cycle_id": self.cycle_id if self.regime == "per_maint" else None,
            "baseline_start": base_start,
            "baseline_end": base_end,
            "train_phases": sorted(train_phases),
            "test_phases": sorted(test_phases),
        }

        if self.regime == "single":
            if w1_start <= base_end:
                raise ValueError(
                    "Single regime test interval is empty: baseline_end is after W1 start. "
                    f"baseline_end={base_end}, W1_start={w1_start}"
                )
            test_start = base_end
            test_end = w1_start
            test_time_mask = (index > test_start) & (index < test_end)
            train_time_mask = baseline_mask
            info.update(
                {
                    "post_train_start": None,
                    "post_train_end": None,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        elif self.regime == "per_maint":
            if self.cycle_id < 0 or self.cycle_id > len(windows):
                raise ValueError(
                    f"cycle_id out of range: got {self.cycle_id}, expected 0..{len(windows)}"
                )
            if self.cycle_id == 0:
                if w1_start <= base_end:
                    raise ValueError(
                        "Per-maint cycle_id=0 (pre_W1) test interval is empty: baseline_end is after W1 start. "
                        f"baseline_end={base_end}, W1_start={w1_start}"
                    )
                test_start = base_end
                test_end = w1_start
                train_time_mask = baseline_mask
                test_time_mask = (index > test_start) & (index < test_end)
                info.update(
                    {
                        "maintenance_id": "pre_W1",
                        "maintenance_start": None,
                        "maintenance_end": None,
                        "post_train_start": None,
                        "post_train_end": None,
                        "test_start": test_start,
                        "test_end": test_end,
                    }
                )
            else:
                j = self.cycle_id - 1
                wj_start, wj_end, wid, _sev = windows[j]
                is_last = j == len(windows) - 1
                next_start = windows[j + 1][0] if not is_last else pd.to_datetime(index.max())
                next_start = pd.to_datetime(next_start)

                post_train_start = pd.to_datetime(wj_end)
                post_train_end = post_train_start + pd.Timedelta(minutes=float(self.post_train_minutes))
                if post_train_end > next_start:
                    post_train_end = next_start

                after_start = post_train_end
                after_end = next_start

                post_train_time_mask = (index >= post_train_start) & (index < post_train_end)
                if is_last:
                    after_time_mask = (index >= after_start) & (index <= after_end)
                else:
                    after_time_mask = (index >= after_start) & (index < after_end)

                train_time_mask = baseline_mask | post_train_time_mask
                test_time_mask = after_time_mask

                info.update(
                    {
                        "maintenance_id": wid,
                        "maintenance_start": pd.to_datetime(wj_start),
                        "maintenance_end": pd.to_datetime(wj_end),
                        "post_train_start": post_train_start,
                        "post_train_end": post_train_end,
                        "test_start": after_start,
                        "test_end": after_end,
                    }
                )
        else:
            raise ValueError(
                f"Unsupported regime={self.regime!r}. Use 'single' or 'per_maint'."
            )

        train_mask = pd.Series(train_time_mask, index=index) & op_phase.isin(train_phases)
        test_mask = pd.Series(test_time_mask, index=index) & op_phase.isin(test_phases)

        # Track raw vs filtered counts for logging/debugging.
        info.update(
            {
                "baseline_rows_time": int(pd.Series(baseline_mask, index=index).sum()),
                "baseline_rows_train_phase": int((pd.Series(baseline_mask, index=index) & op_phase.isin(train_phases)).sum()),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_phase0_rows": int((test_mask & op_phase.eq(0)).sum()),
                "test_phase1_rows": int((test_mask & op_phase.eq(1)).sum()),
                "test_phase2_rows": int((test_mask & op_phase.eq(2)).sum()),
            }
        )
        self.split_info = info

        if train_mask.sum() <= 0:
            raise ValueError("Training mask produced zero rows after phase filtering.")
        if test_mask.sum() <= 0:
            raise ValueError("Test mask produced zero rows after phase filtering.")

        return train_mask.astype(bool), test_mask.astype(bool)

    def setup(self, stage: Optional[str] = None) -> None:
        df_raw = load_csv(self.data_path, self.timestamp_col, drop_unnamed=self.drop_unnamed_index)

        base_feats = select_numeric_features(df_raw, prefer=LIKELY_METROPT_FEATURES)
        if not base_feats:
            raise ValueError("No numeric features found in MetroPT input data.")

        df_base = df_raw[base_feats].copy()
        X = build_rolling_features(df_base, rolling_window=self.rolling_window)
        self.n_features = int(X.shape[1])
        self.base_feature_names = list(base_feats)
        self.rolling_feature_names = [str(col) for col in X.columns]
        self.feature_hash = build_feature_hash(self.rolling_feature_names)

        windows = self._default_windows()
        op_phase = build_operation_phase(
            index=X.index, windows=windows, pre_minutes=self.pre_maint_minutes
        ).astype(np.int8)

        train_mask, test_mask = self._build_masks(X.index, op_phase, windows)

        X_vals = X.to_numpy(dtype=np.float32, copy=False)
        op_phase_vals = op_phase.to_numpy(dtype=np.int8, copy=False)
        train_mask_arr = train_mask.to_numpy(dtype=bool)
        test_mask_arr = test_mask.to_numpy(dtype=bool)

        train_segments_raw = _segments_from_mask(X_vals, train_mask_arr)
        test_segments_raw = _segments_from_mask(X_vals, test_mask_arr)
        train_phase_segments_raw = _segments_from_mask(op_phase_vals, train_mask_arr)
        test_phase_segments_raw = _segments_from_mask(op_phase_vals, test_mask_arr)
        self.train_segment_metadata = _segment_metadata_from_mask(
            X.index, train_mask_arr, op_phase=op_phase
        )
        self.test_segment_metadata = _segment_metadata_from_mask(
            X.index, test_mask_arr, op_phase=op_phase
        )

        if not train_segments_raw:
            raise ValueError("No contiguous training segments were produced (unexpected).")
        if not test_segments_raw:
            raise ValueError("No contiguous test segments were produced (unexpected).")
        if len(train_segments_raw) != len(train_phase_segments_raw):
            raise ValueError("Training phase segments are not aligned with training signal segments.")
        if len(test_segments_raw) != len(test_phase_segments_raw):
            raise ValueError("Test phase segments are not aligned with test signal segments.")

        train_segments_for_dataset_raw = list(train_segments_raw)
        scaler_fit_segments_raw = list(train_segments_raw)
        raw_validation_segments: List[np.ndarray] = []
        validation_split_report: Dict[str, object] = {
            "policy": self.validation_split_policy,
            "validation_strategy": "window_level_chronological_legacy",
            "validation_rows_excluded_from_scaler": 0,
        }
        if self.validation_split_policy == LEAKAGE_FREE_VALIDATION_SPLIT_POLICY:
            if self._finetune_data_policy_active:
                split_plan = self._resolve_finetune_split_plan(
                    len(train_segments_raw[-1])
                )
                scaler_local_train_rows = int(
                    split_plan["scaler_local_train_rows"]
                )
                scaler_fit_segments_raw = list(train_segments_raw[:-1]) + [
                    train_segments_raw[-1][:scaler_local_train_rows]
                ]
                validation_split_report = {
                    "policy": self.validation_split_policy,
                    "validation_strategy": split_plan["validation_strategy"],
                    "validation_rows_excluded_from_scaler": split_plan[
                        "validation_rows_excluded_from_scaler"
                    ],
                    "unused_boundary_raw_rows": split_plan[
                        "unused_boundary_raw_rows"
                    ],
                    "total_local_rows_excluded_from_scaler": split_plan[
                        "total_local_rows_excluded_from_scaler"
                    ],
                    "sequence_embargo_windows": split_plan[
                        "applied_embargo_windows"
                    ],
                    "early_stopping_eligible": split_plan[
                        "early_stopping_eligible"
                    ],
                    "short_local_fallback_applied": split_plan[
                        "short_local_fallback_applied"
                    ],
                }
            elif self.val_size and float(self.val_size) > 0.0:
                (
                    train_segments_for_dataset_raw,
                    raw_validation_segments,
                    validation_split_report,
                ) = self._split_raw_validation_segment(train_segments_raw)
                scaler_fit_segments_raw = list(train_segments_for_dataset_raw)
            else:
                validation_split_report = {
                    "policy": self.validation_split_policy,
                    "validation_strategy": "disabled_no_validation_fraction",
                    "validation_rows_excluded_from_scaler": 0,
                    "early_stopping_eligible": False,
                }

        scaler, preprocessing_report = fit_metropt_preprocessing_scaler(
            scaler_fit_segments_raw,
            self.rolling_feature_names,
            policy=self.preprocessing_policy,
            binary_feature_names=self.binary_feature_names,
        )
        self.scaler = scaler
        self.preprocessing_report = preprocessing_report
        self.validation_split_report = dict(validation_split_report)

        train_segments = [
            scaler.transform(seg).astype(np.float32)
            for seg in train_segments_for_dataset_raw
        ]
        validation_segments = [
            scaler.transform(segment).astype(np.float32)
            for segment in raw_validation_segments
        ]
        test_segments = [scaler.transform(seg).astype(np.float32) for seg in test_segments_raw]

        train_val_ds = MetroPTSegmentedSequenceDataset(
            train_segments, seq_len=self.seq_len, stride=self.stride
        )
        test_ds = MetroPTSegmentedSequenceDataset(
            test_segments,
            phase_segments=test_phase_segments_raw,
            seq_len=self.seq_len,
            stride=self.stride,
        )

        minimum_windows_before_validation = 1 if validation_segments else 2
        if (
            len(train_val_ds) < minimum_windows_before_validation
            and self.val_size > 0
        ):
            raise ValueError(
                f"Not enough train windows to create a validation split: "
                f"train_windows={len(train_val_ds)}, val_size={self.val_size}%"
            )
        if len(test_ds) < 1:
            raise ValueError(
                f"Not enough test windows: test_windows={len(test_ds)} (seq_len={self.seq_len})."
            )

        # Search workflows require an informative PdM test interval. Corrected
        # fine-tuning is unsupervised, so it can still adapt on local healthy
        # windows when a later cycle has no phase-1 test windows; provenance
        # records that its final-training PdM objective is non-informative.
        expects_positive_phase = 1 in set(int(p) for p in self.test_phases)
        if (
            self.regime == "per_maint"
            and int(self.cycle_id) > 0
            and expects_positive_phase
            and int(test_ds.window_positive_count) <= 0
        ):
            if self._finetune_data_policy_active:
                self.split_info["test_informative_for_pdm_objective"] = False
                Log.warning(
                    "FINETUNE_NON_INFORMATIVE_TEST "
                    f"cycle_id={self.cycle_id:02d} test_positive_windows=0; "
                    "continuing because unsupervised fine-tuning only requires local healthy windows."
                )
            else:
                raise ValueError(
                    "Test mask produced zero positive windows after phase filtering "
                    "(non_informative_cycle_no_positive_windows)."
                )
        else:
            self.split_info["test_informative_for_pdm_objective"] = True

        self.split_info.update(
            {
                "n_features": self.n_features,
                "base_feature_names": list(self.base_feature_names),
                "rolling_feature_names": list(self.rolling_feature_names),
                "rolling_aggregations": list(self.rolling_aggregations),
                "feature_hash": self.feature_hash,
                "preprocessing_policy": self.preprocessing_policy,
                "preprocessing_policy_version": preprocessing_report.get("policy_version"),
                "preprocessing_report": dict(preprocessing_report),
                "validation_split_policy": self.validation_split_policy,
                "validation_split_report": dict(validation_split_report),
                "shuffle_train": self._train_shuffle,
                "drop_last_train": self._train_drop_last,
                "train_shuffle_seed": self._train_shuffle_seed,
                "train_segments": int(len(train_segments)),
                "test_segments": int(len(test_segments)),
                "train_segment_metadata": self.train_segment_metadata,
                "test_segment_metadata": self.test_segment_metadata,
                "test_window_label_policy": "end_anchor_phase",
                "test_label_pos_windows": int(test_ds.window_positive_count),
                "test_label_neg_windows": int(test_ds.window_negative_count),
            }
        )

        if self._finetune_data_policy_active:
            self._build_finetune_datasets(train_segments)
        elif validation_segments:
            self.train_dataset = train_val_ds
            self.val_dataset = MetroPTSegmentedSequenceDataset(
                validation_segments,
                seq_len=self.seq_len,
                stride=self.stride,
            )
        elif self.val_size and float(self.val_size) > 0.0:
            total = len(train_val_ds)
            val_windows = max(1, int(np.floor(total * (float(self.val_size) / 100.0))))
            if val_windows >= total:
                val_windows = total - 1
            train_windows = total - val_windows
            train_idx = range(0, train_windows)
            val_idx = range(train_windows, total)
            self.train_dataset = Subset(train_val_ds, list(train_idx))
            self.val_dataset = Subset(train_val_ds, list(val_idx))
        else:
            self.train_dataset = train_val_ds
            self.val_dataset = None

        self.test_dataset = test_ds

        # Log once per datamodule instance to avoid repeated spam from trainer setup cycles.
        if not self._summary_logged:
            Log.info(
                "DATALOADER_SUMMARY "
                f"dataset={self.dataset_name} regime={self.split_info.get('regime')} cycle_id={self.split_info.get('cycle_id')} "
                f"n_features={self.n_features} seq_len={self.seq_len} stride={self.stride} rolling_window={self.rolling_window} "
                f"preprocessing_policy={self.preprocessing_policy} "
                f"validation_split_policy={self.validation_split_policy} "
                f"validation_rows_excluded_from_scaler={validation_split_report.get('validation_rows_excluded_from_scaler', 0)} "
                f"binary_passthrough_features={preprocessing_report.get('passthrough_feature_count', 0)} "
                f"train_rows={self.split_info.get('train_rows')} test_rows={self.split_info.get('test_rows')} "
                f"test_phase0_rows={self.split_info.get('test_phase0_rows')} "
                f"test_phase1_rows={self.split_info.get('test_phase1_rows')} "
                f"test_phase2_rows={self.split_info.get('test_phase2_rows')} "
                f"train_segments={self.split_info.get('train_segments')} test_segments={self.split_info.get('test_segments')} "
                f"train_windows={int(len(self.train_dataset)) if self.train_dataset is not None else 0} "
                f"val_windows={int(len(self.val_dataset)) if self.val_dataset is not None else 0} "
                f"test_windows={int(len(self.test_dataset)) if self.test_dataset is not None else 0} "
                f"test_label_pos_windows={self.split_info.get('test_label_pos_windows')} "
                f"test_label_neg_windows={self.split_info.get('test_label_neg_windows')} "
                f"test_window_label_policy={self.split_info.get('test_window_label_policy')}"
            )
            self._summary_logged = True

    def train_dataloader(self):
        if not self.train_dataset:
            Log.warning("Train dataset is empty. Returning an empty DataLoader.")
            return self._empty_dataloader()
        persistent = bool(self.persistent_workers and self.num_workers > 0)
        drop_last = bool(
            self._train_drop_last and len(self.train_dataset) >= self.batch_size
        )
        generator = None
        if self._train_shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._train_shuffle_seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self._train_shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=generator,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            Log.warning("Validation dataset is empty. Returning an empty DataLoader.")
            return self._empty_dataloader()
        persistent = bool(self.persistent_workers and self.num_workers > 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            Log.warning("Test dataset is empty. Returning an empty DataLoader.")
            return self._empty_dataloader()
        persistent = bool(self.persistent_workers and self.num_workers > 0)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
