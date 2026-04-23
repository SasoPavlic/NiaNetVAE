from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import nianetvae.search.cycle_warmstart as cycle_warmstart
import nianetvae.search.objective_engine as objective_engine
import nianetvae.search.runner as runner_module
import nianetvae.search.winner_selection as winner_selection
from nianetvae.models.rnn_vae import RNNVAE


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
            "pdm": {
                "metric": "fixed_theta_fbeta_covpen",
                "fixed_theta": 0.61,
                "beta": 2.0,
                "coverage_target": 0.20,
                "coverage_penalty_lambda": 0.50,
            },
            "selection": {
                "method": "weighted_ideal_distance",
                "weights": {"error": 0.30, "efficiency": 0.20, "pdm": 0.50},
            },
        },
        "logging_params": {"save_dir": "logs/tests/"},
    }


def _pdm_anomaly_payload(
    *,
    precision: float = 1.0,
    recall: float = 1.0,
    coverage: float = 0.2,
    metric_valid: bool = True,
    invalid_reason: str | None = None,
) -> dict:
    return {
        "pdm_metric_valid": metric_valid,
        "pdm_metric_invalid_reason": invalid_reason,
        "pdm_fixed_theta": 0.61,
        "pdm_beta": 2.0,
        "pdm_coverage_target": 0.20,
        "pdm_coverage_penalty_lambda": 0.50,
        "pdm_fixed_theta_precision": precision,
        "pdm_fixed_theta_recall": recall,
        "pdm_fixed_theta_coverage": coverage,
    }


def test_objective_bundle_uses_selected_raw_error_metric():
    model = _TinyModel()
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"RMSE": 2.75},
        anomaly_metrics=_pdm_anomaly_payload(precision=0.40, recall=0.40, coverage=0.20),
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="RMSE"),
    )

    assert bundle["valid"] is True
    assert bundle["obj_error"] == pytest.approx(2.75)
    assert bundle["obj_efficiency"] > 0


def test_objective_bundle_computes_obj_pdm_from_fixed_theta_formula():
    model = _TinyModel()
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics=_pdm_anomaly_payload(precision=0.50, recall=0.80, coverage=0.30),
        seq_len=16,
        n_features=4,
        cfg=_cfg(),
    )

    assert bundle["valid"] is True
    assert bundle["pdm_signal_quality"] == pytest.approx(0.6517857, abs=1e-6)
    assert bundle["obj_pdm"] == pytest.approx(0.3482143, abs=1e-6)


def test_objective_bundle_uses_worst_case_when_fixed_theta_diagnostics_missing():
    model = _TinyModel()
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics={},
        seq_len=16,
        n_features=4,
        cfg=_cfg(),
    )

    assert bundle["valid"] is True
    assert bundle["reason"] is None
    assert bundle["pdm_signal_quality"] == pytest.approx(0.0)
    assert bundle["obj_pdm"] == pytest.approx(1.0)


def test_efficiency_params_backend_returns_finite_positive():
    model = _TinyModel()
    value, reason = objective_engine._compute_efficiency_objective(
        model=model,
        metric_name="params",
        seq_len=16,
        n_features=4,
    )

    assert reason is None
    assert value is not None
    assert value > 0


def test_efficiency_macs_backend_falls_back_to_params_when_unavailable(monkeypatch):
    model = _TinyModel()
    monkeypatch.setattr(
        objective_engine,
        "_estimate_model_macs",
        lambda *args, **kwargs: (None, "macs_backend_unavailable"),
    )
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics=_pdm_anomaly_payload(precision=0.5, recall=0.5, coverage=0.5),
        seq_len=16,
        n_features=4,
        cfg=_cfg(efficiency_metric="macs"),
    )

    assert bundle["valid"] is True
    assert bundle["reason"] is None
    assert bundle["obj_efficiency"] > 0


def test_efficiency_latency_backend_penalizes_with_reason(monkeypatch):
    model = _TinyModel()
    monkeypatch.setattr(
        objective_engine,
        "_estimate_model_latency_ms",
        lambda *args, **kwargs: (None, "latency_backend_unavailable"),
    )
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics=_pdm_anomaly_payload(precision=0.5, recall=0.5, coverage=0.5),
        seq_len=16,
        n_features=4,
        cfg=_cfg(efficiency_metric="latency_ms"),
    )

    assert bundle["valid"] is False
    assert bundle["obj_efficiency"] == objective_engine.DEFAULT_PENALTY
    assert bundle["reason"] == "latency_backend_unavailable"


def test_problem_initializes_with_three_objectives(tmp_path):
    class _DummyConn:
        def get_entries(self, hash_id, dataset_name):
            return pd.DataFrame()

    ctx = runner_module.SearchRuntimeContext(
        run_uuid="test-run",
        config={**_cfg(), "logging_params": {"save_dir": str(tmp_path)}},
        conn=_DummyConn(),
        datamodule=SimpleNamespace(),
        dataset_name="MetroPT_cycle00",
    )
    runner = runner_module.SearchRunner(ctx)
    problem = runner_module.RNNVAEArchitectureMultiObj(
        dimension=RNNVAE.GENE_DIMENSION,
        runner=runner,
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
                        "obj_error": 1.5,
                        "obj_efficiency": 10.0,
                        "obj_pdm": 0.4,
                        "MAE": 0.1,
                        "MSE": 0.2,
                        "RMSE": 0.3,
                        "MAPE": 0.4,
                        "RMAPE": 0.5,
                        "SMAPE": 1.5,
                        "window_auprc": 0.6,
                        "objective_pdm_metric": "fixed_theta_fbeta_covpen",
                        "window_roc_auc": 0.7,
                        "ranking_metric_valid": True,
                        "ranking_metric_invalid_reason": None,
                        "window_count": 10,
                        "positive_window_count": 2,
                        "negative_window_count": 8,
                        "positive_window_rate": 0.2,
                        "window_reconstruction_error_min": 0.1,
                        "window_reconstruction_error_max": 1.0,
                        "window_reconstruction_error_mean": 0.5,
                        "window_reconstruction_error_std": 0.1,
                        "segment_count": 1,
                        "best_f1_threshold": 0.4,
                        "best_f1_precision": 0.5,
                        "best_f1_recall": 1.0,
                        "best_f1_score": 0.6667,
                        "pdm_fixed_theta": 0.61,
                        "pdm_beta": 2.0,
                        "pdm_coverage_target": 0.2,
                        "pdm_coverage_penalty_lambda": 0.5,
                        "pdm_fixed_theta_precision": 0.5,
                        "pdm_fixed_theta_recall": 1.0,
                        "pdm_fixed_theta_fbeta": 0.8333,
                        "pdm_fixed_theta_coverage": 0.4,
                        "pdm_coverage_excess": 0.25,
                        "pdm_quality_raw": 0.7083,
                        "pdm_quality_clipped": 0.7083,
                        "pdm_metric_valid": True,
                        "pdm_metric_invalid_reason": None,
                    }
                ]
            )

        def save_model_and_entry(self, **kwargs):
            return None

    monkeypatch.setattr(runner_module, "RNNVAE", _CachedModel)
    cfg = {**_cfg(), "logging_params": {"save_dir": str(tmp_path)}}
    ctx = runner_module.SearchRuntimeContext(
        run_uuid="test-run",
        config=cfg,
        conn=_DummyConn(),
        datamodule=SimpleNamespace(),
        dataset_name="MetroPT_cycle00",
    )
    runner = runner_module.SearchRunner(ctx)
    problem = runner_module.RNNVAEArchitectureMultiObj(
        dimension=RNNVAE.GENE_DIMENSION,
        runner=runner,
    )

    out = {}
    problem._evaluate(np.zeros((2, RNNVAE.GENE_DIMENSION), dtype=np.float32), out)

    assert "F" in out
    assert out["F"].shape == (2, 3)
    assert np.isfinite(out["F"]).all()


def test_cached_objective_bundle_requires_contract_provenance_match():
    model = _TinyModel()
    cfg = _cfg()
    cached_row = {
        "SMAPE": 1.0,
        "obj_error": 1.0,
        "obj_efficiency": 10.0,
        "obj_pdm": 0.2,
        "objective_pdm_metric": None,
        "pdm_fixed_theta": 0.61,
        "pdm_beta": 2.0,
        "pdm_coverage_target": 0.2,
        "pdm_coverage_penalty_lambda": 0.5,
        "pdm_metric_valid": True,
        "pdm_fixed_theta_precision": 0.8,
        "pdm_fixed_theta_recall": 0.8,
        "pdm_fixed_theta_coverage": 0.2,
    }

    bundle = objective_engine.calculate_objective_bundle_from_cached_row(
        model=model,
        cached_row=cached_row,
        seq_len=16,
        n_features=4,
        cfg=cfg,
    )

    assert bundle["valid"] is False
    assert bundle["reason"].startswith("cached_objective_contract_mismatch")


def test_warm_start_sampling_uses_effective_population(monkeypatch, tmp_path):
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({"solution": [0.25] * RNNVAE.GENE_DIMENSION}), encoding="utf-8")
    weights_path = tmp_path / "model.pt"
    weights_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        cycle_warmstart,
        "_find_latest_trained_cycle_artifacts_before",
        lambda cycle_id, config, run_uuid=None: (cycle_id - 1, tmp_path, weights_path, meta_path),
    )

    cfg = {
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
        "logging_params": {"save_dir": str(tmp_path)},
    }

    out = cycle_warmstart._resolve_warm_start_sampling(
        dimensionality=RNNVAE.GENE_DIMENSION,
        effective_population=21,
        config=cfg,
        run_uuid="test-run",
    )

    assert out["enabled"] is True
    assert out["sampling"].shape == (21, RNNVAE.GENE_DIMENSION)
    details = out["details"]
    assert details["population_size"] == 21
    assert details["carry_over_count"] + details["perturb_count"] + details["random_count"] == 21


def test_warm_start_cycle0_remains_random_init(monkeypatch):
    cfg = {
        "nia_search": {"warm_start": {"enabled": True}},
        "data_params": {"regime": "per_maint", "cycle_id": 0},
        "exp_params": {"manual_seed": 42},
    }

    out = cycle_warmstart._resolve_warm_start_sampling(
        dimensionality=RNNVAE.GENE_DIMENSION,
        effective_population=21,
        config=cfg,
        run_uuid="test-run",
    )

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
                        "solution_array": json.dumps([0.5] * RNNVAE.GENE_DIMENSION),
                        "obj_error": 1.2,
                        "obj_efficiency": 100.0,
                        "obj_pdm": 0.2,
                        "algorithm_name": "NSGA3",
                        "timestamp": "2026-04-02 12:00:00",
                    }
                ]
            )

    def _fake_minimize(problem, algorithm, termination, seed, verbose, n_jobs):
        captured["algorithm"] = algorithm
        captured["termination"] = termination
        return SimpleNamespace()

    monkeypatch.setattr(runner_module, "NSGA3", _DummyAlgorithm)
    monkeypatch.setattr(runner_module, "minimize", _fake_minimize)
    monkeypatch.setattr(
        runner_module,
        "get_reference_directions",
        lambda name, n_obj, n_partitions: np.zeros((21, 3), dtype=float),
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_warm_start_sampling",
        lambda dimensionality, effective_population, config, run_uuid=None: {
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
    monkeypatch.setattr(
        runner_module,
        "_run_final_training",
        lambda best_solution, **kwargs: {
            "model": SimpleNamespace(hash_id="winner-hash", state_dict=lambda: {}),
            "started_at": "2026-04-02T12:00:00",
            "ended_at": "2026-04-02T12:00:01",
            "duration_s": 1.0,
            "obj_error": 1.2,
            "obj_efficiency": 100.0,
            "obj_pdm": 0.2,
            "pdm_signal_quality": 0.8,
            "objective_reason": None,
            "objective_contract": objective_engine._resolve_objective_contract(_cfg()),
            "metrics": {},
            "anomaly_metrics": _pdm_anomaly_payload(precision=1.0, recall=1.0, coverage=0.2),
        },
    )
    monkeypatch.setattr(runner_module.torch, "save", lambda *args, **kwargs: None)

    cfg = {
        "nia_search": {"time": "00:00:05", "metrics": ["SMAPE"], "nsga3": {"n_partitions": 5}},
        "exp_params": {"manual_seed": 42},
        "data_params": {"cycle_id": 0, "regime": "per_maint"},
        "logging_params": {"save_dir": str(tmp_path), "export_enabled": False},
        "objectives": _cfg()["objectives"],
    }
    ctx = runner_module.SearchRuntimeContext(
        run_uuid="test-run",
        config=cfg,
        conn=_DummyConn(),
        datamodule=SimpleNamespace(),
        dataset_name="MetroPT_cycle00",
    )
    runner = runner_module.SearchRunner(ctx)
    runner.solve_architecture_problem()

    assert isinstance(captured["algorithm"], _DummyAlgorithm)
    assert captured["ref_dirs"].shape == (21, 3)


def test_select_deterministic_pareto_winner_prefers_lower_pdm_on_tie():
    df = pd.DataFrame(
        [
                {
                    "id": 10,
                    "hash_id": "a",
                    "solution_array": json.dumps([0.1] * RNNVAE.GENE_DIMENSION),
                    "obj_error": 1.0,
                    "obj_efficiency": 10.0,
                    "obj_pdm": 0.30,
                    "algorithm_name": "NSGA3",
                    "timestamp": "2026-04-02 12:00:10",
                },
                {
                    "id": 11,
                    "hash_id": "b",
                    "solution_array": json.dumps([0.2] * RNNVAE.GENE_DIMENSION),
                    "obj_error": 2.0,
                    "obj_efficiency": 10.0,
                    "obj_pdm": 0.20,  # better
                    "algorithm_name": "NSGA3",
                    "timestamp": "2026-04-02 12:00:11",
            },
        ]
    )

    tie_contract = {
        "method": "weighted_ideal_distance",
        "weights": {"error": 0.5, "efficiency": 0.0, "pdm": 0.5},
        "weights_normalized": {"error": 0.5, "efficiency": 0.0, "pdm": 0.5},
    }
    selected = winner_selection._select_deterministic_pareto_winner(
        candidates_df=df,
        selection_contract=tie_contract,
        dataset_name="MetroPT_cycle00",
        penalty=objective_engine.DEFAULT_PENALTY,
    )

    assert selected["selected_hash"] == "b"
    assert selected["pareto_candidate_count"] == 2


def test_select_deterministic_pareto_winner_deduplicates_by_hash():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "hash_id": "dup",
                "solution_array": json.dumps([0.1] * RNNVAE.GENE_DIMENSION),
                "obj_error": 2.0,
                "obj_efficiency": 20.0,
                "obj_pdm": 0.50,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:01",
            },
            {
                "id": 2,
                "hash_id": "dup",
                "solution_array": json.dumps([0.2] * RNNVAE.GENE_DIMENSION),
                "obj_error": 1.0,
                "obj_efficiency": 15.0,
                "obj_pdm": 0.30,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:02",
            },
            {
                "id": 3,
                "hash_id": "other",
                "solution_array": json.dumps([0.3] * RNNVAE.GENE_DIMENSION),
                "obj_error": 1.2,
                "obj_efficiency": 16.0,
                "obj_pdm": 0.35,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:03",
            },
        ]
    )

    selected = winner_selection._select_deterministic_pareto_winner(
        candidates_df=df,
        selection_contract=winner_selection._resolve_winner_selection_contract(_cfg()),
        dataset_name="MetroPT_cycle00",
        penalty=objective_engine.DEFAULT_PENALTY,
    )

    assert selected["deduplicated_candidate_count"] == 2
    assert selected["selected_hash"] in {"dup", "other"}


def test_select_deterministic_pareto_winner_fails_fast_on_empty_valid_pool():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "hash_id": "bad",
                "solution_array": json.dumps([0.1] * RNNVAE.GENE_DIMENSION),
                "obj_error": float(objective_engine.DEFAULT_PENALTY),
                "obj_efficiency": float(objective_engine.DEFAULT_PENALTY),
                "obj_pdm": float(objective_engine.DEFAULT_PENALTY),
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-02 12:00:00",
            }
        ]
    )

    with pytest.raises(ValueError, match="no valid objective candidates"):
        winner_selection._select_deterministic_pareto_winner(
            candidates_df=df,
            selection_contract=winner_selection._resolve_winner_selection_contract(_cfg()),
            dataset_name="MetroPT_cycle00",
            penalty=objective_engine.DEFAULT_PENALTY,
        )


def test_select_deterministic_pareto_winner_filters_postgres_real_rounded_penalty():
    # Postgres REAL (float4) can round 9e10 -> 89999998976.0.
    rounded_penalty = float(np.float32(objective_engine.DEFAULT_PENALTY))
    assert rounded_penalty < float(objective_engine.DEFAULT_PENALTY)

    df = pd.DataFrame(
        [
            {
                "id": 1,
                "hash_id": "rounded-penalty",
                "solution_array": json.dumps([0.1] * RNNVAE.GENE_DIMENSION),
                "obj_error": rounded_penalty,
                "obj_efficiency": rounded_penalty,
                "obj_pdm": rounded_penalty,
                "algorithm_name": "NSGA3",
                "timestamp": "2026-04-03 11:16:45",
            }
        ]
    )

    with pytest.raises(ValueError, match="no valid objective candidates"):
        winner_selection._select_deterministic_pareto_winner(
            candidates_df=df,
            selection_contract=winner_selection._resolve_winner_selection_contract(_cfg()),
            dataset_name="MetroPT_cycle20",
            penalty=objective_engine.DEFAULT_PENALTY,
        )
