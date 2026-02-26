from __future__ import annotations

"""
MetroPT-3 DataLoader for NiaNetVAE.

This loader mirrors the MetroPT PdM framework's feature engineering and maintenance-cycle semantics:
  - Rolling aggregation (mean/median/std/skew/min/max) over a configurable time-based window.
  - Maintenance context via Davari et al. (2021) default failure windows and operation_phase labels:
      0 = normal, 1 = pre-maintenance, 2 = maintenance.

Two regimes are supported:
  - regime="single"
      Train: baseline = first train_minutes from dataset start (inclusive cutoff), phases in train_phases.
      Test:  (baseline_end, start_W1), phases in test_phases (typically {0}).
  - regime="per_maint"
      cycle_id=0 maps to pre_W1 (baseline-trained model tested on baseline_end..W1_start).
      cycle_id=1..21 maps to Davari window order (#1..#21).
      post_train = [end_j, min(end_j + post_train_minutes, start_{j+1}))
      after_maint = [end(post_train), start_{j+1})   (or until end of data if j is the last window)
      Train: baseline ∪ post_train, phases in train_phases (typically {0,1}).
      Test:  after_maint, phases in test_phases (typically {0}).

An explicit goal is to prevent sequence windows from crossing gaps between disjoint time blocks. To achieve this,
we build windows within each contiguous True-run of the selected masks (baseline/post-train/test) only.
"""

import os
import platform
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

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
    agg = rolled.aggregate(["mean", "median", "std", "skew", "min", "max"])
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


class MetroPTSegmentedSequenceDataset(Dataset):
    """Sliding-window dataset over multiple contiguous segments (no cross-gap windows)."""

    def __init__(self, segments: List[np.ndarray], seq_len: int = 200, stride: int = 1) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        self._segments: List[np.ndarray] = []
        self._windows_per_segment: List[int] = []

        for seg in segments or []:
            arr = np.asarray(seg, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValueError("Each segment must be a 1D or 2D array.")
            self._segments.append(arr)
            n = arr.shape[0]
            w = max(0, (n - self.seq_len) // self.stride + 1)
            self._windows_per_segment.append(int(w))

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
        return {"signal": signal, "target": 0, "ts_id": seg_idx}


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
        test_phases: Optional[Sequence[int]] = (0,),
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
        self.test_phases = _as_int_list(test_phases, default=(0,))

        self.n_features: Optional[int] = None
        self.split_info: Dict[str, object] = {}
        self._summary_logged = False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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

        windows = self._default_windows()
        op_phase = build_operation_phase(
            index=X.index, windows=windows, pre_minutes=self.pre_maint_minutes
        ).astype(np.int8)

        train_mask, test_mask = self._build_masks(X.index, op_phase, windows)

        X_vals = X.to_numpy(dtype=np.float32, copy=False)
        train_mask_arr = train_mask.to_numpy(dtype=bool)
        test_mask_arr = test_mask.to_numpy(dtype=bool)

        train_segments_raw = _segments_from_mask(X_vals, train_mask_arr)
        test_segments_raw = _segments_from_mask(X_vals, test_mask_arr)

        if not train_segments_raw:
            raise ValueError("No contiguous training segments were produced (unexpected).")
        if not test_segments_raw:
            raise ValueError("No contiguous test segments were produced (unexpected).")

        scaler = StandardScaler()
        for seg in train_segments_raw:
            if seg.shape[0] > 0:
                scaler.partial_fit(seg)

        train_segments = [scaler.transform(seg).astype(np.float32) for seg in train_segments_raw]
        test_segments = [scaler.transform(seg).astype(np.float32) for seg in test_segments_raw]

        train_val_ds = MetroPTSegmentedSequenceDataset(
            train_segments, seq_len=self.seq_len, stride=self.stride
        )
        test_ds = MetroPTSegmentedSequenceDataset(
            test_segments, seq_len=self.seq_len, stride=self.stride
        )

        if len(train_val_ds) < 2 and self.val_size > 0:
            raise ValueError(
                f"Not enough train windows to create a validation split: "
                f"train_windows={len(train_val_ds)}, val_size={self.val_size}%"
            )
        if len(test_ds) < 1:
            raise ValueError(
                f"Not enough test windows: test_windows={len(test_ds)} (seq_len={self.seq_len})."
            )

        self.split_info.update(
            {
                "n_features": self.n_features,
                "train_segments": int(len(train_segments)),
                "test_segments": int(len(test_segments)),
            }
        )

        if self.val_size and float(self.val_size) > 0.0:
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
                f"train_rows={self.split_info.get('train_rows')} test_rows={self.split_info.get('test_rows')} "
                f"train_segments={self.split_info.get('train_segments')} test_segments={self.split_info.get('test_segments')} "
                f"train_windows={int(len(self.train_dataset)) if self.train_dataset is not None else 0} "
                f"val_windows={int(len(self.val_dataset)) if self.val_dataset is not None else 0} "
                f"test_windows={int(len(self.test_dataset)) if self.test_dataset is not None else 0}"
            )
            self._summary_logged = True

    def train_dataloader(self):
        if not self.train_dataset:
            Log.warning("Train dataset is empty. Returning an empty DataLoader.")
            return self._empty_dataloader()
        persistent = bool(self.persistent_workers and self.num_workers > 0)
        drop_last = bool(len(self.train_dataset) >= self.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
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
