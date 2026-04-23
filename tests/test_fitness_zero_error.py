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


def _objective_cfg(error_metric: str = "SMAPE", efficiency_metric: str = "params") -> dict:
    return {
        "data_params": {"n_features": 3},
        "objectives": {
            "error": {"metric": error_metric},
            "efficiency": {"metric": efficiency_metric},
            "pdm": {
                "metric": "fixed_theta_fbeta_covpen",
                "fixed_theta": 0.61,
                "beta": 2.0,
                "coverage_target": 0.20,
                "coverage_penalty_lambda": 0.50,
            },
        },
    }


def _pdm_payload(precision: float, recall: float, coverage: float) -> dict:
    return {
        "pdm_metric_valid": True,
        "pdm_metric_invalid_reason": None,
        "pdm_fixed_theta": 0.61,
        "pdm_beta": 2.0,
        "pdm_coverage_target": 0.20,
        "pdm_coverage_penalty_lambda": 0.50,
        "pdm_fixed_theta_precision": precision,
        "pdm_fixed_theta_recall": recall,
        "pdm_fixed_theta_coverage": coverage,
    }


def test_objective_bundle_zero_error_is_valid():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 0.0}),
        anomaly_metrics=_pdm_payload(precision=0.7, recall=0.7, coverage=0.2),
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
    assert bundle["obj_efficiency"] > 0
    assert bundle["obj_error"] != DEFAULT_PENALTY
    assert bundle["obj_pdm"] != DEFAULT_PENALTY


def test_objective_bundle_uses_raw_smape_value():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 1.23456789}),
        anomaly_metrics=_pdm_payload(precision=1.0, recall=1.0, coverage=0.2),
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
    assert bundle["obj_efficiency"] > 0
    assert bundle["obj_pdm"] == pytest.approx(0.0)


def test_objective_bundle_penalizes_when_smape_missing():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"MAE": 0.1}),
        anomaly_metrics=_pdm_payload(precision=0.75, recall=0.75, coverage=0.2),
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
    assert bundle["obj_efficiency"] == DEFAULT_PENALTY
    assert bundle["obj_pdm"] == DEFAULT_PENALTY
