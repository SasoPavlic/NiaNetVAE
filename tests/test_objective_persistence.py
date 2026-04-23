from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.search.runtime_artifacts import _export_cycle_artifacts
from nianetvae.storage.experiment_storage import SQLiteConnector


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.hash_id = "dummy-hash"
        self.activation_name = "Tanh"
        self.optimizer_name = "Adam"
        self.encoder_layer_step = 16
        self.encoder_num_layers = 2
        self.decoder_num_layers = 2
        self.decoder_layer_step = 16
        self.encoding_layers = [3, 2]
        self.decoding_layers = [2, 3]
        self.bottleneck_size = 2
        self.mapping_context = {"source": "test"}


class _DummyMetrics:
    MAE = 0.1
    MSE = 0.2
    RMSE = 0.3
    MAPE = 0.4
    RMAPE = 0.5
    SMAPE = 1.25


def test_sqlite_objective_only_schema_and_insert(tmp_path: Path):
    db_path = tmp_path / "objective_only.sqlite"
    connector = SQLiteConnector(str(db_path), "solutions_s1_t7")

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(solutions_s1_t7)").fetchall()}
        assert {"obj_error", "obj_efficiency", "obj_pdm"}.issubset(columns)
        assert {
            "window_auprc",
            "window_roc_auc",
            "ranking_metric_valid",
            "ranking_metric_invalid_reason",
            "window_count",
            "positive_window_count",
            "negative_window_count",
            "positive_window_rate",
            "window_reconstruction_error_min",
            "window_reconstruction_error_max",
            "window_reconstruction_error_mean",
            "window_reconstruction_error_std",
            "segment_count",
            "best_f1_threshold",
            "best_f1_precision",
            "best_f1_recall",
            "best_f1_score",
            "pdm_fixed_theta",
            "pdm_beta",
            "pdm_coverage_target",
            "pdm_coverage_penalty_lambda",
            "pdm_fixed_theta_precision",
            "pdm_fixed_theta_recall",
            "pdm_fixed_theta_fbeta",
            "pdm_fixed_theta_coverage",
            "pdm_coverage_excess",
            "pdm_quality_raw",
            "pdm_quality_clipped",
            "pdm_metric_valid",
            "pdm_metric_invalid_reason",
            "objective_pdm_metric",
        }.issubset(columns)
        assert "error" not in columns
        assert "complexity" not in columns
        assert "fitness" not in columns
        assert "_".join(("pr", "auc", "mean")) not in columns
        assert "_".join(("roc", "auc", "mean")) not in columns
        assert "precision" not in columns
        assert "recall" not in columns
        assert "f1_score" not in columns
    finally:
        conn.close()

    model = _DummyModel()
    experiment = SimpleNamespace(
        metrics=_DummyMetrics(),
        anomaly_metrics={
            "window_auprc": 0.67,
            "window_roc_auc": 0.72,
            "ranking_metric_valid": True,
            "ranking_metric_invalid_reason": None,
            "window_count": 100,
            "positive_window_count": 10,
            "negative_window_count": 90,
            "positive_window_rate": 0.1,
            "window_reconstruction_error_min": 0.01,
            "window_reconstruction_error_max": 1.5,
            "window_reconstruction_error_mean": 0.4,
            "window_reconstruction_error_std": 0.2,
            "segment_count": 2,
            "best_f1_threshold": 0.5,
            "best_f1_precision": 0.3,
            "best_f1_recall": 0.8,
            "best_f1_score": 0.4364,
            "pdm_fixed_theta": 0.61,
            "pdm_beta": 2.0,
            "pdm_coverage_target": 0.2,
            "pdm_coverage_penalty_lambda": 0.5,
            "pdm_fixed_theta_precision": 0.3,
            "pdm_fixed_theta_recall": 0.8,
            "pdm_fixed_theta_fbeta": 0.6250,
            "pdm_fixed_theta_coverage": 0.1,
            "pdm_coverage_excess": 0.0,
            "pdm_quality_raw": 0.6250,
            "pdm_quality_clipped": 0.6250,
            "pdm_metric_valid": True,
            "pdm_metric_invalid_reason": None,
        },
    )
    connector.save_model_and_entry(
        dataset_name="MetroPT_cycle00",
        alg_name="NSGA3",
        iteration=1,
        solution=np.asarray([0.1] * 6, dtype=float),
        model=model,
        experiment=experiment,
        obj_error=1.25,
        obj_efficiency=1234.0,
        obj_pdm=0.33,
        objective_contract={
            "pdm_metric": "fixed_theta_fbeta_covpen",
        },
    )

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT obj_error, obj_efficiency, obj_pdm, window_auprc, window_roc_auc, "
            "ranking_metric_valid, window_count, positive_window_count, best_f1_score, "
            "pdm_quality_clipped, objective_pdm_metric "
            "FROM solutions_s1_t7 LIMIT 1"
        ).fetchone()
        assert row is not None
        assert float(row[0]) == 1.25
        assert float(row[1]) == 1234.0
        assert float(row[2]) == 0.33
        assert float(row[3]) == 0.67
        assert float(row[4]) == 0.72
        assert bool(row[5]) is True
        assert int(row[6]) == 100
        assert int(row[7]) == 10
        assert float(row[8]) == 0.4364
        assert float(row[9]) == 0.625
        assert str(row[10]) == "fixed_theta_fbeta_covpen"
    finally:
        conn.close()

    df = connector.get_cycle_candidates("MetroPT_cycle00", algorithm_name="NSGA3")
    assert {"obj_error", "obj_efficiency", "obj_pdm"}.issubset(set(df.columns))
    assert "error" not in set(df.columns)
    assert "complexity" not in set(df.columns)
    assert "fitness" not in set(df.columns)


def test_sqlite_schema_mismatch_auto_migrates_for_old_anomaly_columns(tmp_path: Path):
    db_path = tmp_path / "old_schema.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        old_auprc_column = "_".join(("pr", "auc", "mean"))
        conn.execute(
            f"""
            CREATE TABLE solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT,
                {old_auprc_column} REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    SQLiteConnector(str(db_path), "solutions")
    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(solutions)").fetchall()}
        assert "objective_pdm_metric" in columns
        assert "pdm_fixed_theta" in columns
        assert "pdm_metric_valid" in columns
    finally:
        conn.close()


def test_export_artifacts_include_objective_and_selection_provenance(tmp_path: Path):
    model = _DummyModel()
    export_dir = tmp_path / "cycle_00"
    config = {
        "data_params": {
            "dataset_name": "MetroPT",
            "regime": "per_maint",
            "cycle_id": 0,
            "n_features": 90,
            "seq_len": 200,
        },
        "workflow": {"mode": "per_maint_warmstart_search"},
        "exp_params": {"manual_seed": 42, "optimizer": "Adam", "learning_rate": 0.003, "weight_decay": 0.0},
        "logging_params": {},
    }
    search_result = {
        "iterations": 3,
        "trained": 3,
        "cached": 0,
        "invalid": 0,
        "failed": 0,
        "selected_distance": 0.1234,
        "winner_selection": {
            "method": "weighted_ideal_distance",
            "weights_normalized": {"error": 0.3, "efficiency": 0.2, "pdm": 0.5},
            "selected_hash": "dummy-hash",
            "selected_objectives": {
                "obj_error": 1.25,
                "obj_efficiency": 1234.0,
                "obj_pdm": 0.33,
            },
            "selected_distance": 0.1234,
        },
    }
    final_result = {
        "experiment": type(
            "_ExperimentStub",
            (),
            {"learning_rate": 0.003, "weight_decay": 0.0},
        )(),
        "started_at": datetime(2026, 4, 3, 9, 0, 0),
        "ended_at": datetime(2026, 4, 3, 9, 0, 5),
        "duration_s": 5.0,
        "obj_error": 1.25,
        "obj_efficiency": 1234.0,
        "obj_pdm": 0.33,
        "pdm_signal_quality": 0.67,
        "objective_reason": None,
        "objective_contract": {
            "error_metric": "SMAPE",
            "efficiency_metric": "macs",
            "pdm_metric": "fixed_theta_fbeta_covpen",
            "pdm_fixed_theta": 0.61,
            "pdm_beta": 2.0,
            "pdm_coverage_target": 0.2,
            "pdm_coverage_penalty_lambda": 0.5,
        },
        "metrics": {"SMAPE": 1.25},
        "anomaly_metrics": {
            "window_auprc": 0.67,
            "pdm_fixed_theta": 0.61,
            "pdm_fixed_theta_precision": 0.7,
            "pdm_fixed_theta_recall": 0.8,
            "pdm_fixed_theta_coverage": 0.3,
            "pdm_quality_clipped": 0.67,
            "pdm_metric_valid": True,
        },
    }

    _, meta_path, summary_path = _export_cycle_artifacts(
        export_dir=export_dir,
        model=model,
        best_solution=np.asarray([0.1] * 6, dtype=float),
        best_algorithm="NSGA3",
        search_result=search_result,
        final_result=final_result,
        config=config,
        dataset_name="MetroPT_cycle00",
        run_uuid="run-uuid-test",
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert meta["winner_selection"]["method"] == "weighted_ideal_distance"
    assert meta["winner_selection"]["selected_objectives"]["obj_pdm"] == 0.33
    assert meta["training_policy"]["optimizer"] == "Adam"
    assert meta["training_policy"]["learning_rate"] == 0.003
    assert meta["training_policy"]["weight_decay"] == 0.0
    assert meta["final_training_anomaly_metrics"]["window_auprc"] == 0.67

    final_training = summary["final_training"]
    assert final_training["training_policy"]["optimizer"] == "Adam"
    assert final_training["obj_error"] == 1.25
    assert final_training["obj_efficiency"] == 1234.0
    assert final_training["obj_pdm"] == 0.33
    assert "error" not in final_training
    assert "complexity" not in final_training
    assert "fitness" not in final_training
    assert final_training["anomaly_metrics"]["window_auprc"] == 0.67


def test_experiment_uses_fixed_adam_training_policy():
    model = _DummyModel()
    experiment = RNNVAExperiment(
        model,
        dataset_name="MetroPT_cycle00",
        alg_name="NSGA3",
        exp_params={"optimizer": "Adam", "learning_rate": 0.003, "weight_decay": 0.0, "kld_weight": 0.1},
        data_params={"seq_len": 200, "n_features": 90},
    )

    optimizer = experiment.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.003
    assert optimizer.defaults["weight_decay"] == 0.0
