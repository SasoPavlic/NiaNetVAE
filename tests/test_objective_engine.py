from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import nianetvae.rnn_vae_architecture_search as search


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)
        self.encoding_layers = [3]
        self.decoding_layers = [4]
        self.bottleneck_size = 3

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["signal"]
        else:
            x = batch
        return {"signal": x, "reconstructed": x}


class _DummyMetrics:
    def __init__(self, payload):
        self._payload = dict(payload)

    def are_metrics_complete(self):
        return True

    def compute(self):
        return dict(self._payload)


def _cfg(error_metric: str = "SMAPE", efficiency_metric: str = "params") -> dict:
    return {
        "data_params": {
            "dataset_name": "MetroPT",
            "regime": "per_maint",
            "seq_len": 16,
            "n_features": 4,
        },
        "objectives": {
            "error": {"metric": error_metric},
            "efficiency": {"metric": efficiency_metric},
            "pdm": {"metric": "auprc_premaint"},
            "selection": {
                "method": "weighted_ideal_distance",
                "weights": {"error": 0.30, "efficiency": 0.20, "pdm": 0.50},
            },
        },
        "logging_params": {"save_dir": "logs/tests/"},
    }


def test_objective_bundle_uses_selected_raw_error_metric():
    model = _TinyModel()
    bundle = search.calculate_objective_bundle(
        model=model,
        metrics_payload={"RMSE": 2.75},
        anomaly_metrics={"pr_auc_mean": 0.40},
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="RMSE"),
    )

    assert bundle["valid"] is True
    assert bundle["obj_error"] == pytest.approx(2.75)
    assert bundle["obj_efficiency"] > 0
    assert bundle["fitness"] == pytest.approx(bundle["obj_error"] + bundle["obj_efficiency"])


def test_objective_bundle_computes_obj_pdm_from_pr_auc():
    model = _TinyModel()
    bundle = search.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics={"pr_auc_mean": 0.83},
        seq_len=16,
        n_features=4,
        cfg=_cfg(),
    )

    assert bundle["valid"] is True
    assert bundle["pdm_signal_quality"] == pytest.approx(0.83)
    assert bundle["obj_pdm"] == pytest.approx(0.17)


def test_objective_bundle_penalizes_missing_pr_auc():
    model = _TinyModel()
    bundle = search.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics={},
        seq_len=16,
        n_features=4,
        cfg=_cfg(),
    )

    assert bundle["valid"] is False
    assert bundle["obj_pdm"] == search.PENALTY
    assert bundle["reason"] == "missing_or_invalid_pdm_signal_quality"


def test_efficiency_params_backend_returns_finite_positive():
    model = _TinyModel()
    value, reason = search._compute_efficiency_objective(
        model=model,
        metric_name="params",
        seq_len=16,
        n_features=4,
    )

    assert reason is None
    assert value is not None
    assert value > 0


def test_efficiency_macs_backend_penalizes_with_reason(monkeypatch):
    model = _TinyModel()
    monkeypatch.setattr(
        search,
        "_estimate_model_macs",
        lambda *args, **kwargs: (None, "macs_backend_unavailable"),
    )
    bundle = search.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics={"pr_auc_mean": 0.5},
        seq_len=16,
        n_features=4,
        cfg=_cfg(efficiency_metric="macs"),
    )

    assert bundle["valid"] is False
    assert bundle["obj_efficiency"] == search.PENALTY
    assert bundle["reason"] == "macs_backend_unavailable"


def test_efficiency_latency_backend_penalizes_with_reason(monkeypatch):
    model = _TinyModel()
    monkeypatch.setattr(
        search,
        "_estimate_model_latency_ms",
        lambda *args, **kwargs: (None, "latency_backend_unavailable"),
    )
    bundle = search.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics={"pr_auc_mean": 0.5},
        seq_len=16,
        n_features=4,
        cfg=_cfg(efficiency_metric="latency_ms"),
    )

    assert bundle["valid"] is False
    assert bundle["obj_efficiency"] == search.PENALTY
    assert bundle["reason"] == "latency_backend_unavailable"


def test_problem_initializes_with_three_objectives(tmp_path):
    class _DummyConn:
        def get_entries(self, hash_id, dataset_name):
            return pd.DataFrame()

    problem = search.RNNVAEArchitectureMultiObj(
        dimension=7,
        config={**_cfg(), "logging_params": {"save_dir": str(tmp_path)}},
        conn=_DummyConn(),
        datamodule=SimpleNamespace(),
        dataset_name="MetroPT_cycle00",
    )

    assert problem.n_obj == 3


def test_cached_evaluation_emits_three_objective_values(monkeypatch, tmp_path):
    class _CachedModel(torch.nn.Module):
        def __init__(self, solution, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.hash_id = "cached-hash"
            self.is_valid = True
            self.encoding_layers = [2]
            self.decoding_layers = [2]
            self.bottleneck_size = 2

        def get_hash(self):
            return self.hash_id

        def forward(self, batch):
            if isinstance(batch, dict):
                x = batch["signal"]
            else:
                x = batch
            return {"signal": x, "reconstructed": x}

    class _DummyConn:
        def get_entries(self, hash_id, dataset_name):
            return pd.DataFrame(
                [
                    {
                        "hash_id": hash_id,
                        "dataset_name": dataset_name,
                        "error": 1.5,
                        "complexity": 10.0,
                        "fitness": 11.5,
                        "MAE": 0.1,
                        "MSE": 0.2,
                        "RMSE": 0.3,
                        "MAPE": 0.4,
                        "RMAPE": 0.5,
                        "SMAPE": 1.5,
                        "pr_auc_mean": 0.6,
                        "pr_auc_std": 0.1,
                        "roc_auc_mean": 0.7,
                        "roc_auc_std": 0.1,
                    }
                ]
            )

        def save_model_and_entry(self, **kwargs):
            return None

    monkeypatch.setattr(search, "RNNVAE", _CachedModel)
    cfg = {**_cfg(), "logging_params": {"save_dir": str(tmp_path)}}
    problem = search.RNNVAEArchitectureMultiObj(
        dimension=7,
        config=cfg,
        conn=_DummyConn(),
        datamodule=SimpleNamespace(),
        dataset_name="MetroPT_cycle00",
    )

    out = {}
    problem._evaluate(np.zeros((2, 7), dtype=np.float32), out)

    assert "F" in out
    assert out["F"].shape == (2, 3)
    assert np.isfinite(out["F"]).all()


def test_warm_start_sampling_uses_effective_population(monkeypatch, tmp_path):
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({"solution": [0.25] * 7}), encoding="utf-8")
    weights_path = tmp_path / "model.pt"
    weights_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        search,
        "_find_latest_trained_cycle_artifacts_before",
        lambda cycle_id: (cycle_id - 1, tmp_path, weights_path, meta_path),
    )
    monkeypatch.setattr(
        search,
        "config",
        {
            "nia_search": {
                "warm_start": {
                    "enabled": True,
                    "carry_over_ratio": 0.10,
                    "perturb_ratio": 0.40,
                    "perturbation_strength": 0.08,
                }
            },
            "data_params": {"regime": "per_maint", "cycle_id": 2},
            "exp_params": {"manual_seed": 42},
        },
    )

    out = search._resolve_warm_start_sampling(dimensionality=7, effective_population=21)

    assert out["enabled"] is True
    assert out["sampling"].shape == (21, 7)
    details = out["details"]
    assert details["population_size"] == 21
    assert details["carry_over_count"] + details["perturb_count"] + details["random_count"] == 21


def test_warm_start_cycle0_remains_random_init(monkeypatch):
    monkeypatch.setattr(
        search,
        "config",
        {
            "nia_search": {"warm_start": {"enabled": True}},
            "data_params": {"regime": "per_maint", "cycle_id": 0},
            "exp_params": {"manual_seed": 42},
        },
    )

    out = search._resolve_warm_start_sampling(dimensionality=7, effective_population=21)

    assert out["enabled"] is False
    assert out["init_mode"] == "random"
    assert out["reason"] == "cycle_00_random_init"
    assert out["details"]["random_count"] == 21


def test_solve_architecture_problem_uses_nsga3_ref_dirs(monkeypatch, tmp_path):
    captured = {}

    class _DummyAlgorithm:
        def __init__(self, ref_dirs, sampling=None, **kwargs):
            captured["ref_dirs"] = ref_dirs
            captured["sampling"] = sampling
            self.ref_dirs = ref_dirs

    class _DummyConn:
        def get_cycle_candidates(self, dataset_name, algorithm_name="NSGA3"):
            return pd.DataFrame(
                [
                    {
                        "id": 1,
                        "hash_id": "winner-hash",
                        "solution_array": json.dumps([0.5] * 7),
                        "error": 1.2,
                        "complexity": 100.0,
                        "pr_auc_mean": 0.8,
                        "algorithm_name": "NSGA3",
                        "timestamp": "2026-04-02 12:00:00",
                        "fitness": 101.2,
                    }
                ]
            )

    def _fake_minimize(problem, algorithm, termination, seed, verbose, n_jobs):
        captured["algorithm"] = algorithm
        captured["termination"] = termination
        return SimpleNamespace()

    monkeypatch.setattr(search, "NSGA3", _DummyAlgorithm)
    monkeypatch.setattr(search, "minimize", _fake_minimize)
    monkeypatch.setattr(
        search,
        "get_reference_directions",
        lambda name, n_obj, n_partitions: np.zeros((21, 3), dtype=float),
    )
    monkeypatch.setattr(
        search,
        "_resolve_warm_start_sampling",
        lambda dimensionality, effective_population: {
            "enabled": False,
            "sampling": None,
            "init_mode": "random",
            "source_cycle_id": None,
            "reason": "warm_start_disabled",
            "details": {
                "enabled": False,
                "population_size": effective_population,
                "carry_over_count": 0,
                "perturb_count": 0,
                "random_count": effective_population,
                "perturbation_strength": None,
            },
        },
    )
    monkeypatch.setattr(search, "conn", _DummyConn())
    monkeypatch.setattr(search, "datamodule", SimpleNamespace())
    monkeypatch.setattr(search, "dataset_name", "MetroPT_cycle00")
    monkeypatch.setattr(
        search,
        "_run_final_training",
        lambda best_solution: {
            "model": SimpleNamespace(hash_id="winner-hash", state_dict=lambda: {}),
            "started_at": "2026-04-02T12:00:00",
            "ended_at": "2026-04-02T12:00:01",
            "duration_s": 1.0,
            "fitness": 101.2,
            "error": 1.2,
            "complexity": 100.0,
            "obj_pdm": 0.2,
            "pdm_signal_quality": 0.8,
            "objective_reason": None,
            "objective_contract": search._resolve_objective_contract(_cfg()),
            "metrics": {},
            "anomaly_metrics": {"pr_auc_mean": 0.8},
        },
    )
    monkeypatch.setattr(search.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        search,
        "config",
        {
            "nia_search": {"time": "00:00:05", "metrics": ["SMAPE"], "nsga3": {"n_partitions": 5}},
            "exp_params": {"manual_seed": 42},
            "data_params": {"cycle_id": 0, "regime": "per_maint"},
            "logging_params": {"save_dir": str(tmp_path), "export_enabled": False},
        },
    )

    search.solve_architecture_problem()

    assert isinstance(captured["algorithm"], _DummyAlgorithm)
    assert captured["ref_dirs"].shape == (21, 3)


def test_select_deterministic_pareto_winner_prefers_lower_pdm_on_tie():
    df = pd.DataFrame(
        [
                {
                    "id": 10,
                    "hash_id": "a",
                    "solution_array": json.dumps([0.1] * 7),
                    "error": 1.0,
                    "complexity": 10.0,
                    "pr_auc_mean": 0.70,  # obj_pdm=0.30
                    "algorithm_name": "NSGA3",
                    "timestamp": "2026-04-02 12:00:10",
                    "fitness": 11.0,
                },
                {
                    "id": 11,
                    "hash_id": "b",
                    "solution_array": json.dumps([0.2] * 7),
                    "error": 2.0,
                    "complexity": 10.0,
                    "pr_auc_mean": 0.80,  # obj_pdm=0.20 (better)
                    "algorithm_name": "NSGA3",
                    "timestamp": "2026-04-02 12:00:11",
                    "fitness": 11.0,
            },
        ]
    )

    tie_contract = {
        "method": "weighted_ideal_distance",
        "weights": {"error": 0.5, "efficiency": 0.0, "pdm": 0.5},
        "weights_normalized": {"error": 0.5, "efficiency": 0.0, "pdm": 0.5},
    }
    selected = search._select_deterministic_pareto_winner(
        candidates_df=df,
        selection_contract=tie_contract,
    )

    assert selected["selected_hash"] == "b"
    assert selected["pareto_candidate_count"] == 2


def test_select_deterministic_pareto_winner_deduplicates_by_hash():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "hash_id": "dup",
                "solution_array": json.dumps([0.1] * 7),
                "error": 2.0,
                "complexity": 20.0,
                "pr_auc_mean": 0.50,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:01",
                "fitness": 22.0,
            },
            {
                "id": 2,
                "hash_id": "dup",
                "solution_array": json.dumps([0.2] * 7),
                "error": 1.0,
                "complexity": 15.0,
                "pr_auc_mean": 0.70,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:02",
                "fitness": 16.0,
            },
            {
                "id": 3,
                "hash_id": "other",
                "solution_array": json.dumps([0.3] * 7),
                "error": 1.2,
                "complexity": 16.0,
                "pr_auc_mean": 0.65,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:03",
                "fitness": 17.2,
            },
        ]
    )

    selected = search._select_deterministic_pareto_winner(
        candidates_df=df,
        selection_contract=search._resolve_winner_selection_contract(_cfg()),
    )

    assert selected["deduplicated_candidate_count"] == 2
    assert selected["selected_hash"] in {"dup", "other"}


def test_select_deterministic_pareto_winner_fails_fast_on_empty_valid_pool():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "hash_id": "bad",
                "solution_array": json.dumps([0.1] * 7),
                "error": float(search.PENALTY),
                "complexity": float(search.PENALTY),
                "pr_auc_mean": None,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:00",
                "fitness": float(search.PENALTY),
            }
        ]
    )

    with pytest.raises(ValueError, match="no valid objective candidates"):
        search._select_deterministic_pareto_winner(
            candidates_df=df,
            selection_contract=search._resolve_winner_selection_contract(_cfg()),
        )
