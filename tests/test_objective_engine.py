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
                "metric": "smoothed_rank_gap",
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
    smoothed_auroc: float = 0.8,
    positive_risk_mean: float | None = None,
    negative_risk_mean: float | None = None,
    metric_valid: bool = True,
    invalid_reason: str | None = None,
) -> dict:
    if positive_risk_mean is not None and negative_risk_mean is not None:
        smoothed_rank_gap = float(positive_risk_mean) - float(negative_risk_mean)
        smoothed_auroc = 0.5 * (smoothed_rank_gap + 1.0)
    smoothed_rank_gap = 2.0 * float(smoothed_auroc) - 1.0
    return {
        "pdm_metric_valid": metric_valid,
        "pdm_metric_invalid_reason": invalid_reason,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": 0.8,
        "pdm_negative_smoothed_risk_mean": 0.2,
        "pdm_smoothed_auroc": smoothed_auroc,
        "pdm_smoothed_rank_gap": smoothed_rank_gap,
    }


def test_objective_bundle_uses_selected_raw_error_metric():
    model = _TinyModel()
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"RMSE": 2.75},
        anomaly_metrics=_pdm_anomaly_payload(smoothed_auroc=0.6),
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="RMSE"),
    )

    assert bundle["valid"] is True
    assert bundle["obj_error"] == pytest.approx(2.75)
    assert bundle["obj_efficiency"] > 0


def test_objective_bundle_computes_obj_pdm_from_smoothed_rank_gap_formula():
    model = _TinyModel()
    bundle = objective_engine.calculate_objective_bundle(
        model=model,
        metrics_payload={"SMAPE": 1.0},
        anomaly_metrics=_pdm_anomaly_payload(smoothed_auroc=0.85),
        seq_len=16,
        n_features=4,
        cfg=_cfg(),
    )

    assert bundle["valid"] is True
    assert bundle["pdm_signal_quality"] == pytest.approx(0.7, abs=1e-6)
    assert bundle["obj_pdm"] == pytest.approx(0.15, abs=1e-6)


def test_objective_bundle_uses_worst_case_when_smoothed_rank_diagnostics_missing():
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
    assert bundle["pdm_signal_quality"] is None
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
        anomaly_metrics=_pdm_anomaly_payload(),
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
        anomaly_metrics=_pdm_anomaly_payload(),
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


@pytest.mark.parametrize("metric_key", ["SMAPE", "smape"])
def test_cached_objective_bundle_accepts_case_insensitive_error_metric(metric_key):
    model = _TinyModel()
    cached_row = {
        metric_key: 1.25,
        "objective_pdm_metric": "smoothed_rank_gap",
        "pdm_metric_valid": True,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": 0.8,
        "pdm_negative_smoothed_risk_mean": 0.2,
        "pdm_smoothed_auroc": 0.8,
        "pdm_smoothed_rank_gap": 0.6,
    }

    bundle = objective_engine.calculate_objective_bundle_from_cached_row(
        model=model,
        cached_row=cached_row,
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="SMAPE"),
    )

    assert bundle["valid"] is True
    assert bundle["obj_error"] == pytest.approx(1.25)
    assert bundle["obj_pdm"] == pytest.approx(0.2)


def test_cached_objective_bundle_prefers_exact_metric_key_over_lowercase_fallback():
    model = _TinyModel()
    cached_row = {
        "SMAPE": 1.25,
        "smape": 9.99,
        "objective_pdm_metric": "smoothed_rank_gap",
        "pdm_metric_valid": True,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": 0.8,
        "pdm_negative_smoothed_risk_mean": 0.2,
        "pdm_smoothed_auroc": 0.8,
        "pdm_smoothed_rank_gap": 0.6,
    }

    bundle = objective_engine.calculate_objective_bundle_from_cached_row(
        model=model,
        cached_row=cached_row,
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="SMAPE"),
    )

    assert bundle["valid"] is True
    assert bundle["obj_error"] == pytest.approx(1.25)


def test_cached_objective_bundle_rejects_missing_error_metric():
    model = _TinyModel()
    cached_row = {
        "MAE": 1.25,
        "objective_pdm_metric": "smoothed_rank_gap",
        "pdm_metric_valid": True,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": 0.8,
        "pdm_negative_smoothed_risk_mean": 0.2,
        "pdm_smoothed_auroc": 0.8,
        "pdm_smoothed_rank_gap": 0.6,
    }

    bundle = objective_engine.calculate_objective_bundle_from_cached_row(
        model=model,
        cached_row=cached_row,
        seq_len=16,
        n_features=4,
        cfg=_cfg(error_metric="SMAPE"),
    )

    assert bundle["valid"] is False
    assert bundle["reason"] == "missing_or_invalid_error_metric:SMAPE"


def test_cached_evaluation_uses_db_then_memory_cache(monkeypatch, tmp_path):
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
        def __init__(self):
            self.get_entries_calls = 0
            self.save_calls = 0

        def get_entries(self, hash_id, dataset_name):
            self.get_entries_calls += 1
            return pd.DataFrame(
                [
                    {
                        "hash_id": hash_id,
                        "dataset_name": dataset_name,
                        "obj_error": 1.5,
                        "obj_efficiency": 10.0,
                        "obj_pdm": 0.4,
                        "mae": 0.1,
                        "mse": 0.2,
                        "rmse": 0.3,
                        "mape": 0.4,
                        "rmape": 0.5,
                        "smape": 1.5,
                        "objective_pdm_metric": "smoothed_rank_gap",
                        "window_count": 10,
                        "positive_window_count": 2,
                        "negative_window_count": 8,
                        "positive_window_rate": 0.2,
                        "window_reconstruction_error_min": 0.1,
                        "window_reconstruction_error_max": 1.0,
                        "window_reconstruction_error_mean": 0.5,
                        "window_reconstruction_error_std": 0.1,
                        "segment_count": 1,
                        "pdm_smoothing_window_windows": 480,
                        "pdm_positive_smoothed_risk_mean": 0.7,
                        "pdm_negative_smoothed_risk_mean": 0.5,
                        "pdm_smoothed_auroc": 0.6,
                        "pdm_smoothed_rank_gap": 0.2,
                        "pdm_metric_valid": True,
                        "pdm_metric_invalid_reason": None,
                    }
                ]
            )

        def save_model_and_entry(self, **kwargs):
            self.save_calls += 1
            return None

    monkeypatch.setattr(runner_module, "RNNVAE", _CachedModel)
    cfg = {**_cfg(), "logging_params": {"save_dir": str(tmp_path)}}
    conn = _DummyConn()
    ctx = runner_module.SearchRuntimeContext(
        run_uuid="test-run",
        config=cfg,
        conn=conn,
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
    assert out["F"][0].tolist() == pytest.approx(out["F"][1].tolist())
    assert conn.get_entries_calls == 1
    assert conn.save_calls == 0
    assert problem.stats["cached"] == 2
    assert problem.stats["cached_db"] == 1
    assert problem.stats["cached_memory"] == 1
    assert problem.stats["cache_miss"] == 0


def test_duplicate_hash_trains_once_then_uses_memory_cache(monkeypatch, tmp_path):
    class _TrainableModel(torch.nn.Module):
        def __init__(self, solution, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.hash_id = "duplicate-hash"
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

    class _DummyExperiment:
        def __init__(self, *args, **kwargs):
            self.metrics = _DummyMetrics({"SMAPE": 1.25})
            self.anomaly_metrics = _pdm_anomaly_payload(
                positive_risk_mean=0.7,
                negative_risk_mean=0.5,
            )

        def collect_calibration_scores(self, *args, **kwargs):
            return None

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            return None

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return None

    class _DummyDataModule:
        def train_dataloader(self):
            return []

    class _EmptyConn:
        def __init__(self):
            self.get_entries_calls = 0
            self.save_calls = 0

        def get_entries(self, hash_id, dataset_name):
            self.get_entries_calls += 1
            return pd.DataFrame()

        def save_model_and_entry(self, **kwargs):
            self.save_calls += 1
            return None

    monkeypatch.setattr(runner_module, "RNNVAE", _TrainableModel)
    monkeypatch.setattr(runner_module, "RNNVAExperiment", _DummyExperiment)
    monkeypatch.setattr(runner_module, "Trainer", _DummyTrainer)

    cfg = {
        **_cfg(),
        "logging_params": {"save_dir": str(tmp_path)},
        "trainer_params": {},
    }
    conn = _EmptyConn()
    ctx = runner_module.SearchRuntimeContext(
        run_uuid="test-run",
        config=cfg,
        conn=conn,
        datamodule=_DummyDataModule(),
        dataset_name="MetroPT_cycle00",
    )
    runner = runner_module.SearchRunner(ctx)
    problem = runner_module.RNNVAEArchitectureMultiObj(
        dimension=RNNVAE.GENE_DIMENSION,
        runner=runner,
    )

    out = {}
    problem._evaluate(np.zeros((2, RNNVAE.GENE_DIMENSION), dtype=np.float32), out)

    assert out["F"].shape == (2, 3)
    assert out["F"][0].tolist() == pytest.approx(out["F"][1].tolist())
    assert conn.get_entries_calls == 1
    assert conn.save_calls == 1
    assert problem.stats["trained"] == 1
    assert problem.stats["cached"] == 1
    assert problem.stats["cached_memory"] == 1
    assert problem.stats["cached_db"] == 0


def test_cached_objective_bundle_requires_contract_provenance_match():
    model = _TinyModel()
    cfg = _cfg()
    cached_row = {
        "SMAPE": 1.0,
        "obj_error": 1.0,
        "obj_efficiency": 10.0,
        "obj_pdm": 0.2,
        "objective_pdm_metric": None,
        "pdm_metric_valid": True,
        "pdm_smoothing_window_windows": 480,
        "pdm_positive_smoothed_risk_mean": 0.8,
        "pdm_negative_smoothed_risk_mean": 0.2,
        "pdm_smoothed_auroc": 0.8,
        "pdm_smoothed_rank_gap": 0.6,
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
            "anomaly_metrics": _pdm_anomaly_payload(positive_risk_mean=1.0, negative_risk_mean=0.0),
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
