"""Small, reproducible figures derived only from exported workflow tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_workflow_timeline(
    predictions: pd.DataFrame,
    *,
    events: Sequence,
    selected_theta: float,
    coverage_percent: float,
    output: str | Path,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    display = (
        frame[["maintenance_risk", "alarm"]]
        .resample("30min")
        .agg({"maintenance_risk": "mean", "alarm": "max"})
    )
    figure, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    axis.plot(
        display.index,
        display["maintenance_risk"],
        color="#255f85",
        linewidth=0.9,
        label="120-minute maintenance risk",
    )
    axis.axhline(
        selected_theta,
        color="#c43c39",
        linestyle="--",
        linewidth=1.2,
        label=f"Selected threshold ({selected_theta:.3f})",
    )
    active = display["alarm"].fillna(False).astype(bool)
    axis.fill_between(
        display.index,
        0,
        1,
        where=active,
        transform=axis.get_xaxis_transform(),
        color="#e48b26",
        alpha=0.16,
        label=f"Alarm-active time (coverage {coverage_percent:.2f}%)",
    )
    for event in events:
        axis.axvline(pd.Timestamp(event.start), color="#222222", alpha=0.25, linewidth=0.7)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Maintenance risk")
    axis.set_xlabel("MetroPT timestamp")
    axis.set_title("Selected operating point on the shared evaluation timeline")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right", fontsize=8, ncol=2)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_theta_tradeoff(
    sweep: pd.DataFrame,
    *,
    selected_theta: float,
    output: str | Path,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.plot(sweep["coverage_percent"], sweep["recall"], marker="o", markersize=3, label="Recall")
    axis.plot(sweep["coverage_percent"], sweep["f1"], marker="s", markersize=3, label="F1")
    selected = sweep.loc[np.isclose(sweep["maintenance_risk_theta"], selected_theta)]
    if not selected.empty:
        row = selected.iloc[0]
        axis.scatter(
            [row["coverage_percent"]],
            [row["recall"]],
            color="#c43c39",
            s=70,
            zorder=4,
            label=f"Selected theta={selected_theta:.3f}",
        )
    axis.set_xlabel("Alarm coverage (%)")
    axis.set_ylabel("Event metric")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend()
    axis.set_title("Retrospective threshold trade-off")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_workflow_comparison(frame: pd.DataFrame, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = frame["workflow_id"].str.replace("_", " ").tolist()
    positions = np.arange(len(frame))
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    width = 0.25
    for offset, metric in enumerate(("precision", "recall", "f1")):
        axes[0, 0].bar(positions + (offset - 1) * width, frame[metric], width, label=metric.title())
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Event detection")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(positions, frame["coverage_percent"], color="#e48b26")
    axes[0, 1].set_title("Alarm coverage")
    axes[0, 1].set_ylabel("Percent")

    axes[1, 0].bar(positions, frame["ttd_minutes"].fillna(0), color="#3a8d5d")
    axes[1, 0].set_title("Mean time to detection")
    axes[1, 0].set_ylabel("Minutes")

    axes[1, 1].bar(positions, frame["far_per_day"].fillna(0), color="#9a5fb4")
    axes[1, 1].set_title("False alarm intervals per day")
    axes[1, 1].set_ylabel("Intervals/day")

    for axis in axes.flat:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.15)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
