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


def test_window_anomaly_ranking_metrics_compute_global_auc_and_diagnostics():
    targets, predictions = _targets_and_predictions([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    ts_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    metrics = WindowAnomalyRankingMetrics()

    metrics.update(predictions=predictions, targets=targets, labels=labels, ts_ids=ts_ids)
    out = metrics.compute()

    assert out["ranking_metric_valid"] is True
    assert out["ranking_metric_invalid_reason"] is None
    assert out["window_auprc"] == pytest.approx(1.0)
    assert out["window_roc_auc"] == pytest.approx(1.0)
    assert out["window_count"] == 4
    assert out["positive_window_count"] == 2
    assert out["negative_window_count"] == 2
    assert out["positive_window_rate"] == pytest.approx(0.5)
    assert out["window_reconstruction_error_min"] == pytest.approx(0.1)
    assert out["window_reconstruction_error_max"] == pytest.approx(0.9)
    assert out["window_reconstruction_error_mean"] == pytest.approx(0.5)
    assert out["segment_count"] == 2
    assert out["best_f1_threshold"] is not None
    assert out["best_f1_precision"] == pytest.approx(1.0)
    assert out["best_f1_recall"] == pytest.approx(1.0)
    assert out["best_f1_score"] == pytest.approx(1.0)
    assert out["pdm_metric_valid"] is True
    assert out["pdm_metric_invalid_reason"] is None
    assert out["pdm_fixed_theta"] == pytest.approx(0.61)
    assert out["pdm_fixed_theta_precision"] == pytest.approx(1.0)
    assert out["pdm_fixed_theta_recall"] == pytest.approx(1.0)
    assert out["pdm_fixed_theta_coverage"] == pytest.approx(0.5)
    assert out["pdm_fixed_theta_fbeta"] == pytest.approx(1.0)
    assert out["pdm_quality_clipped"] == pytest.approx(0.8125, abs=1e-4)


def test_window_anomaly_ranking_metrics_single_class_is_invalid_with_diagnostics():
    targets, predictions = _targets_and_predictions([0.1, 0.2, 0.3])
    labels = torch.tensor([0, 0, 0], dtype=torch.int64)
    metrics = WindowAnomalyRankingMetrics()

    metrics.update(predictions=predictions, targets=targets, labels=labels)
    out = metrics.compute()

    assert out["ranking_metric_valid"] is False
    assert out["ranking_metric_invalid_reason"] == "no_positive_windows"
    assert out["window_auprc"] is None
    assert out["window_roc_auc"] is None
    assert out["best_f1_threshold"] is None
    assert out["pdm_metric_valid"] is False
    assert out["pdm_metric_invalid_reason"] == "no_positive_windows"
    assert out["window_count"] == 3
    assert out["positive_window_count"] == 0
    assert out["negative_window_count"] == 3
    assert out["segment_count"] == 1


def test_window_anomaly_ranking_metrics_detects_non_finite_scores():
    targets, predictions = _targets_and_predictions([0.1, 0.2, 0.3])
    predictions[1, 0, 0] = float("nan")
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    metrics = WindowAnomalyRankingMetrics()

    metrics.update(predictions=predictions, targets=targets, labels=labels)
    out = metrics.compute()

    assert out["ranking_metric_valid"] is False
    assert out["ranking_metric_invalid_reason"] == "non_finite_scores"
    assert out["window_auprc"] is None
    assert out["window_roc_auc"] is None
