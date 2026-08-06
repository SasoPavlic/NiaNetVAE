from __future__ import annotations

import pandas as pd

from nianetvae.config import EvaluationConfig
from nianetvae.evaluation.event import alarm_intervals, evaluate_maintenance_prediction
from nianetvae.evaluation.risk import (
    build_segmented_maintenance_risk,
    evaluate_risk_thresholds,
    select_operating_point,
    theta_values,
)


def test_event_metrics_and_operating_point_use_one_mask() -> None:
    index = pd.date_range("2020-01-01", periods=240, freq="min")
    evaluation = pd.Series(True, index=index)
    risk = pd.Series(0.0, index=index)
    risk.loc[index[90:121]] = 0.9
    windows = [(index[120], index[125])]
    config = EvaluationConfig(
        risk_window_minutes=10,
        theta_grid="0.10:0.90:0.40",
        extra_thresholds=(),
        target_recall=0.5,
        target_coverage=0.5,
        fixed_lead_minutes=30,
        lead_step_minutes=10,
    )
    rows = evaluate_risk_thresholds(risk, windows, eval_mask=evaluation, config=config)
    selected = select_operating_point(rows, config)
    predictions = risk >= selected["maintenance_risk_theta"]
    metrics = evaluate_maintenance_prediction(
        predictions,
        windows,
        30,
        method_name="test",
        eval_mask=evaluation,
        lead_step_minutes=10,
    )
    assert theta_values(config) == [0.1, 0.5, 0.9]
    assert metrics["event_scores"]["recall"] == 1.0
    assert 0.0 < metrics["coverage"]["alarm_coverage"] < 0.5
    assert "mtia_minutes" in metrics["mtia"]
    assert "first_alarm_accuracy" in metrics["first_alarm_accuracy"]


def test_production_theta_grid_contains_the_frozen_23_values() -> None:
    values = theta_values(EvaluationConfig())
    assert len(values) == 23
    assert values[:3] == [0.1, 0.15, 0.2]
    assert values[-3:] == [0.985, 0.99, 0.995]


def test_maintenance_risk_resets_at_telemetry_gaps() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            "2020-01-01 00:02:00",
            "2020-01-01 00:04:00",
            "2020-01-01 00:05:00",
        ]
    )
    scores = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=index)
    risk = build_segmented_maintenance_risk(
        scores,
        [pd.Series(True, index=index)],
        exceedance_quantile=0.95,
        risk_window_minutes=120,
    )
    assert risk.loc[index[2]] == 1.0
    assert risk.loc[index[3]] == 0.0

    intervals = alarm_intervals(pd.Series(True, index=index))
    assert intervals == [(index[0], index[2]), (index[3], index[4])]
