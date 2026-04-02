from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import nianetvae.rnn_vae_architecture_search as search


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
            "pdm": {"metric": "auprc_premaint"},
        },
    }


def test_calculate_fitness_zero_error_is_valid():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 0.0}),
        anomaly_metrics={"pr_auc_mean": 0.70},
    )
    model = _TinyModel()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=20,
        cfg=_objective_cfg(),
    )

    assert error == pytest.approx(0.0)
    assert complexity > 0
    assert fitness == pytest.approx(error + complexity)
    assert fitness != search.PENALTY


def test_calculate_fitness_uses_raw_smape_value():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 1.23456789}),
        anomaly_metrics={"pr_auc_mean": 0.80},
    )
    model = _TinyModel()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=20,
        cfg=_objective_cfg(),
    )

    assert error == pytest.approx(1.23456789)
    assert complexity > 0
    assert fitness == pytest.approx(error + complexity)


def test_calculate_fitness_penalizes_when_smape_missing():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"MAE": 0.1}),
        anomaly_metrics={"pr_auc_mean": 0.75},
    )
    model = _TinyModel()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=20,
        cfg=_objective_cfg(error_metric="SMAPE"),
    )

    assert fitness == search.PENALTY
    assert error == search.PENALTY
    assert complexity == search.PENALTY
