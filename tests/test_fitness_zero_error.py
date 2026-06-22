from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nianetvae.search.objective_engine import (
    DEFAULT_PENALTY,
    calculate_objective_bundle_from_experiment,
)


class _DummyMetrics:
    def __init__(self, payload):
        self._payload = dict(payload)

    def are_metrics_complete(self):
        return True

    def compute(self):
        return dict(self._payload)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.encoding_layers = [2]
        self.decoding_layers = [3]
        self.bottleneck_size = 2

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["signal"]
        else:
            x = batch
        return {"signal": x, "reconstructed": x}


def _objective_cfg(error_metric: str = "SMAPE") -> dict:
    return {
        "data_params": {"n_features": 3},
        "objectives": {
            "error": {"metric": error_metric},
            "pdm": {
                "metric": "smoothed_rank_gap",
            },
            "alarm_burden": {"metric": "normal_high_risk_rate", "risk_threshold": 0.95},
        },
    }


def _pdm_payload(positive_risk_mean: float, negative_risk_mean: float) -> dict:
    smoothed_rank_gap = float(positive_risk_mean) - float(negative_risk_mean)
    smoothed_auroc = 0.5 * (smoothed_rank_gap + 1.0)
    return {
        "pdm_metric_valid": True,
        "pdm_metric_invalid_reason": None,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": positive_risk_mean,
        "pdm_negative_smoothed_risk_mean": negative_risk_mean,
        "pdm_smoothed_auroc": smoothed_auroc,
        "pdm_smoothed_rank_gap": smoothed_rank_gap,
        "pdm_alarm_burden_threshold": 0.95,
        "pdm_positive_high_risk_rate": 1.0,
        "pdm_negative_high_risk_rate": negative_risk_mean,
    }


def test_objective_bundle_zero_error_is_valid():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 0.0}),
        anomaly_metrics=_pdm_payload(positive_risk_mean=0.7, negative_risk_mean=0.2),
    )
    model = _TinyModel()

    bundle = calculate_objective_bundle_from_experiment(
        model=model,
        experiment=experiment,
        seq_len=20,
        n_features=3,
        cfg=_objective_cfg(),
    )

    assert bundle["obj_error"] == pytest.approx(0.0)
    assert bundle["obj_alarm_burden"] == pytest.approx(0.2)
    assert bundle["diagnostic_params"] > 0
    assert bundle["obj_error"] != DEFAULT_PENALTY
    assert bundle["obj_pdm"] != DEFAULT_PENALTY


def test_objective_bundle_uses_raw_smape_value():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 1.23456789}),
        anomaly_metrics=_pdm_payload(positive_risk_mean=1.0, negative_risk_mean=0.0),
    )
    model = _TinyModel()

    bundle = calculate_objective_bundle_from_experiment(
        model=model,
        experiment=experiment,
        seq_len=20,
        n_features=3,
        cfg=_objective_cfg(),
    )

    assert bundle["obj_error"] == pytest.approx(1.23456789)
    assert bundle["obj_alarm_burden"] == pytest.approx(0.0)
    assert bundle["diagnostic_params"] > 0
    assert bundle["obj_pdm"] == pytest.approx(0.0)


def test_objective_bundle_penalizes_when_smape_missing():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"MAE": 0.1}),
        anomaly_metrics=_pdm_payload(positive_risk_mean=0.75, negative_risk_mean=0.25),
    )
    model = _TinyModel()

    bundle = calculate_objective_bundle_from_experiment(
        model=model,
        experiment=experiment,
        seq_len=20,
        n_features=3,
        cfg=_objective_cfg(error_metric="SMAPE"),
    )

    assert bundle["obj_error"] == DEFAULT_PENALTY
    assert bundle["obj_pdm"] == DEFAULT_PENALTY
    assert bundle["obj_alarm_burden"] == DEFAULT_PENALTY
