from __future__ import annotations

import pandas as pd

from nianetvae.visualization import (
    plot_theta_tradeoff,
    plot_workflow_comparison,
    plot_workflow_timeline,
)


def test_evidence_figures_render_with_explicit_coverage_label(tmp_path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=20, freq="min")
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "maintenance_risk": [value / 19 for value in range(20)],
            "alarm": [value >= 15 for value in range(20)],
        }
    )
    event = type("Event", (), {"start": timestamps[18]})()
    assert plot_workflow_timeline(
        predictions,
        events=[event],
        selected_theta=0.75,
        coverage_percent=25.0,
        output=tmp_path / "timeline.png",
    ).is_file()
    sweep = pd.DataFrame(
        {
            "maintenance_risk_theta": [0.5, 0.75],
            "coverage_percent": [40.0, 25.0],
            "recall": [1.0, 0.8],
            "f1": [0.6, 0.7],
        }
    )
    assert plot_theta_tradeoff(
        sweep, selected_theta=0.75, output=tmp_path / "tradeoff.png"
    ).is_file()
    comparison = pd.DataFrame(
        {
            "workflow_id": ["a", "b"],
            "precision": [0.5, 0.7],
            "recall": [0.8, 0.6],
            "f1": [0.6, 0.65],
            "coverage_percent": [20.0, 15.0],
            "ttd_minutes": [20.0, 25.0],
            "far_per_day": [1.0, 0.5],
        }
    )
    assert plot_workflow_comparison(comparison, tmp_path / "comparison.png").is_file()
