from __future__ import annotations

import pytest
import torch

from nianetvae.experiments.anomaly_evaluation import WindowAnomalyRankingMetrics


def _targets_and_predictions(scores):
    targets = torch.zeros((len(scores), 2, 2), dtype=torch.float32)
    predictions = torch.zeros_like(targets)
    for idx, score in enumerate(scores):
        predictions[idx, :, :] = float(score) ** 0.5
    return targets, predictions


def _compute_with_calibration(test_scores, labels):
    metrics = WindowAnomalyRankingMetrics()
    cal_targets, cal_predictions = _targets_and_predictions([0.1, 0.2, 0.3, 0.4])
    targets, predictions = _targets_and_predictions(test_scores)
    metrics.update_calibration(predictions=cal_predictions, targets=cal_targets)
    metrics.update(
        predictions=predictions,
        targets=targets,
        labels=torch.tensor(labels, dtype=torch.int64),
        ts_ids=torch.arange(len(labels), dtype=torch.int64),
    )
    return metrics.compute()


def test_window_anomaly_ranking_metrics_compute_calibrated_risk_gap_diagnostics():
    out = _compute_with_calibration([0.0, 0.0, 0.5, 0.5], [0, 0, 1, 1])

    assert out["window_count"] == 4
    assert out["positive_window_count"] == 2
    assert out["negative_window_count"] == 2
    assert out["positive_window_rate"] == pytest.approx(0.5)
    assert out["window_reconstruction_error_min"] == pytest.approx(0.0)
    assert out["window_reconstruction_error_max"] == pytest.approx(0.5)
    assert out["segment_count"] == 4
    assert out["calibration_window_count"] == 4
    assert out["risk_score_min"] == pytest.approx(0.0)
    assert out["risk_score_max"] == pytest.approx(1.0)
    assert out["pdm_positive_risk_mean"] == pytest.approx(1.0)
    assert out["pdm_negative_risk_mean"] == pytest.approx(0.0)
    assert out["pdm_risk_gap"] == pytest.approx(1.0)
    assert out["pdm_metric_valid"] is True
    assert out["pdm_metric_invalid_reason"] is None


def test_window_anomaly_ranking_metrics_no_separation_has_zero_gap():
    out = _compute_with_calibration([0.2, 0.2, 0.2, 0.2], [0, 0, 1, 1])

    assert out["pdm_positive_risk_mean"] == pytest.approx(0.5)
    assert out["pdm_negative_risk_mean"] == pytest.approx(0.5)
    assert out["pdm_risk_gap"] == pytest.approx(0.0)


def test_window_anomaly_ranking_metrics_inverted_signal_has_negative_gap():
    out = _compute_with_calibration([0.5, 0.5, 0.0, 0.0], [0, 0, 1, 1])

    assert out["pdm_positive_risk_mean"] == pytest.approx(0.0)
    assert out["pdm_negative_risk_mean"] == pytest.approx(1.0)
    assert out["pdm_risk_gap"] == pytest.approx(-1.0)


def test_window_anomaly_ranking_metrics_single_class_is_invalid_with_diagnostics():
    out = _compute_with_calibration([0.1, 0.2, 0.3], [0, 0, 0])

    assert out["pdm_metric_valid"] is False
    assert out["pdm_metric_invalid_reason"] == "no_positive_windows"
    assert out["window_count"] == 3
    assert out["positive_window_count"] == 0
    assert out["negative_window_count"] == 3
    assert out["segment_count"] == 3


def test_window_anomaly_ranking_metrics_detects_non_finite_scores():
    targets, predictions = _targets_and_predictions([0.1, 0.2, 0.3])
    predictions[1, 0, 0] = float("nan")
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    metrics = WindowAnomalyRankingMetrics()

    cal_targets, cal_predictions = _targets_and_predictions([0.1, 0.2, 0.3])
    metrics.update_calibration(predictions=cal_predictions, targets=cal_targets)
    metrics.update(predictions=predictions, targets=targets, labels=labels)
    out = metrics.compute()

    assert out["pdm_metric_valid"] is False
    assert out["pdm_metric_invalid_reason"] == "non_finite_scores"


def test_window_anomaly_ranking_metrics_missing_calibration_is_invalid():
    targets, predictions = _targets_and_predictions([0.1, 0.2, 0.3])
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    metrics = WindowAnomalyRankingMetrics()

    metrics.update(predictions=predictions, targets=targets, labels=labels)
    out = metrics.compute()

    assert out["pdm_metric_valid"] is False
    assert out["pdm_metric_invalid_reason"] == "missing_calibration_scores"
