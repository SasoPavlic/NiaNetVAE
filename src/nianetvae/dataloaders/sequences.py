"""Gap-safe sequence datasets and common timestamp helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def true_runs(mask: pd.Series) -> list[np.ndarray]:
    aligned = mask.astype(bool).to_numpy()
    positions = np.flatnonzero(aligned)
    if positions.size == 0:
        return []
    boundary_flags = np.diff(positions) > 1
    if isinstance(mask.index, pd.DatetimeIndex) and positions.size > 1:
        full_deltas = np.diff(mask.index.asi8)
        positive_deltas = full_deltas[full_deltas > 0]
        if positive_deltas.size:
            expected = float(np.median(positive_deltas))
            selected_deltas = np.diff(mask.index.asi8[positions])
            boundary_flags |= selected_deltas > expected * 1.5
    boundaries = np.flatnonzero(boundary_flags) + 1
    return [part for part in np.split(positions, boundaries) if part.size]


def contiguous_frames(frame: pd.DataFrame, mask: pd.Series) -> list[pd.DataFrame]:
    aligned = mask.reindex(frame.index).fillna(False).astype(bool)
    return [frame.iloc[positions].copy() for positions in true_runs(aligned)]


def sequence_anchor_mask(mask: pd.Series, sequence_length: int, stride: int = 1) -> pd.Series:
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive.")
    out = pd.Series(False, index=mask.index, dtype=bool)
    for positions in true_runs(mask):
        if positions.size < sequence_length:
            continue
        anchors = positions[sequence_length - 1 :: stride]
        out.iloc[anchors] = True
    return out


class SegmentedSequenceDataset(Dataset):
    """Sequences built independently inside each contiguous input segment."""

    def __init__(
        self,
        segments: Sequence[pd.DataFrame | np.ndarray],
        *,
        sequence_length: int,
        stride: int = 1,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        if self.sequence_length < 1 or self.stride < 1:
            raise ValueError("sequence_length and stride must be positive.")
        arrays: list[np.ndarray] = []
        indices: list[pd.Index | None] = []
        windows: list[tuple[int, int]] = []
        for segment_index, segment in enumerate(segments):
            index = segment.index if isinstance(segment, pd.DataFrame) else None
            values = (
                segment.to_numpy(dtype=np.float32, copy=False)
                if isinstance(segment, pd.DataFrame)
                else np.asarray(segment, dtype=np.float32)
            )
            if values.ndim != 2:
                raise ValueError("Every sequence segment must be a two-dimensional array.")
            arrays.append(values)
            indices.append(index)
            if len(values) >= self.sequence_length:
                for start in range(0, len(values) - self.sequence_length + 1, self.stride):
                    windows.append((segment_index, start))
        self._segments = arrays
        self._indices = indices
        self._windows = windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        segment_index, start = self._windows[item]
        stop = start + self.sequence_length
        window = torch.from_numpy(self._segments[segment_index][start:stop])
        return window, segment_index, stop - 1

    def anchor_index(self, segment_index: int, anchor_offset: int):
        index = self._indices[int(segment_index)]
        if index is None:
            return int(anchor_offset)
        return index[int(anchor_offset)]
