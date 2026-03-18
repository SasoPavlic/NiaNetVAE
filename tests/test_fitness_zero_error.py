from __future__ import annotations

from types import SimpleNamespace

import nianetvae.rnn_vae_architecture_search as search

class _DummyMetrics:
    def __init__(self, payload):
        self._payload = dict(payload)

    def are_metrics_complete(self):
        return True

    def compute(self):
        return dict(self._payload)


def _dummy_model():
    return SimpleNamespace(
        encoding_layers=[object()],
        decoding_layers=[object()],
        bottleneck_size=1,
    )


def test_calculate_fitness_zero_error_is_valid(monkeypatch):
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 0.0}),
        anomaly_metrics={},
    )
    model = _dummy_model()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=200,
    )

    assert error == 0
    assert complexity > 0
    assert fitness == error + complexity
    assert fitness != search.PENALTY


def test_calculate_fitness_uses_raw_smape_value():
    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"SMAPE": 1.23456789}),
        anomaly_metrics={},
    )
    model = _dummy_model()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=200,
    )

    assert error == 1234568
    assert complexity > 0
    assert fitness == error + complexity


def test_calculate_fitness_penalizes_when_smape_missing():

    experiment = SimpleNamespace(
        metrics=_DummyMetrics({"MAE": 0.1}),
        anomaly_metrics={},
    )
    model = _dummy_model()

    fitness, error, complexity = search.calculate_fitness(
        model,
        experiment,
        seq_len=200,
    )

    assert fitness == search.PENALTY
    assert error == search.PENALTY
    assert complexity == search.PENALTY
