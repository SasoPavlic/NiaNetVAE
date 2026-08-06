from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nianetvae.config import (
    AdaptationConfig,
    ArtifactConfig,
    CalibrationConfig,
    DataConfig,
    EvaluationConfig,
    PreprocessingConfig,
    SearchConfig,
    StudyConfig,
    TrainingConfig,
)


def synthetic_config(
    root: Path,
    *,
    workflows: tuple[str, ...] = ("iforest_static",),
    rows: int = 600,
) -> StudyConfig:
    index = pd.date_range("2020-01-01", periods=rows, freq="min")
    frame = pd.DataFrame({"timestamp": index})
    continuous = (
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Motor_current",
        "Oil_temperature",
    )
    binary = (
        "Caudal_impulses",
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
    )
    x = np.arange(rows, dtype=float)
    for offset, name in enumerate(continuous):
        frame[name] = np.sin(x / (13.0 + offset)) + 0.01 * offset
    for offset, name in enumerate(binary):
        frame[name] = ((np.floor(x / (7 + offset)) + offset) % 2).astype(int)
    for center in (240, 430):
        warning = (x >= center - 30) & (x < center)
        frame.loc[warning, list(continuous)] += np.linspace(0.0, 2.0, int(warning.sum()))[:, None]
    source = root / "MetroPT3.csv"
    frame.to_csv(source, index=False)

    windows = (
        ("2020-01-01 04:00:00", "2020-01-01 04:10:00", "#1", "high"),
        ("2020-01-01 07:10:00", "2020-01-01 07:20:00", "#2", "high"),
    )
    return StudyConfig(
        study_name="synthetic controlled test",
        data=DataConfig(
            input_path=str(source),
            timestamp_column="timestamp",
            rolling_window="5min",
            initial_train_minutes=180,
            post_maintenance_train_minutes=30,
            pre_maintenance_minutes=30,
            sequence_length=6,
            stride=1,
            validation_fraction=0.10,
            maintenance_windows=windows,
        ),
        preprocessing=PreprocessingConfig(),
        training=TrainingConfig(
            seed=42,
            batch_size=32,
            min_epochs=1,
            max_epochs=1,
            patience=1,
            num_workers=0,
            device="cpu",
        ),
        adaptation=AdaptationConfig(min_epochs=1, max_epochs=1),
        search=replace(
            SearchConfig(),
            n_partitions=1,
            max_generations=1,
            max_time="00:10:00",
            candidate_min_epochs=1,
            candidate_max_epochs=1,
        ),
        calibration=CalibrationConfig(),
        evaluation=EvaluationConfig(
            risk_window_minutes=10,
            theta_grid="0.10:0.90:0.40",
            extra_thresholds=(),
            target_recall=0.5,
            target_coverage=0.5,
            fixed_lead_minutes=30,
            sensitivity_leads=(10, 20),
            lead_step_minutes=10,
        ),
        artifacts=ArtifactConfig(
            root=str(root / "artifacts"),
            study_id="synthetic_v1",
            save_plots=False,
        ),
        workflows=workflows,
    ).validate()
