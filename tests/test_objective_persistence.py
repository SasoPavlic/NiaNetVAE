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
    connector = SQLiteConnector(str(db_path), "solutions_finetune_riskgap")

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(solutions_finetune_riskgap)").fetchall()}
        assert {"obj_error", "obj_efficiency", "obj_pdm"}.issubset(columns)
        assert {
            "window_count",
            "positive_window_count",
            "negative_window_count",
            "positive_window_rate",
            "window_reconstruction_error_min",
            "window_reconstruction_error_max",
            "window_reconstruction_error_mean",
            "window_reconstruction_error_std",
            "calibration_window_count",
            "calibration_window_reconstruction_error_min",
            "calibration_window_reconstruction_error_max",
            "calibration_window_reconstruction_error_mean",
            "calibration_window_reconstruction_error_std",
            "risk_score_min",
            "risk_score_max",
            "risk_score_mean",
            "risk_score_std",
            "segment_count",
            "pdm_positive_risk_mean",
            "pdm_negative_risk_mean",
            "pdm_risk_gap",
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
        assert "window_auprc" not in columns
        assert "window_roc_auc" not in columns
        assert "ranking_metric_valid" not in columns
        assert "best_f1_score" not in columns
    finally:
        conn.close()

    model = _DummyModel()
    experiment = SimpleNamespace(
        metrics=_DummyMetrics(),
        anomaly_metrics={
            "window_count": 100,
            "positive_window_count": 10,
            "negative_window_count": 90,
            "positive_window_rate": 0.1,
            "window_reconstruction_error_min": 0.01,
            "window_reconstruction_error_max": 1.5,
            "window_reconstruction_error_mean": 0.4,
            "window_reconstruction_error_std": 0.2,
            "calibration_window_count": 20,
            "calibration_window_reconstruction_error_min": 0.01,
            "calibration_window_reconstruction_error_max": 1.0,
            "calibration_window_reconstruction_error_mean": 0.3,
            "calibration_window_reconstruction_error_std": 0.1,
            "risk_score_min": 0.05,
            "risk_score_max": 1.0,
            "risk_score_mean": 0.4,
            "risk_score_std": 0.2,
            "segment_count": 2,
            "pdm_positive_risk_mean": 0.8,
            "pdm_negative_risk_mean": 0.2,
            "pdm_risk_gap": 0.6,
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
            "pdm_metric": "calibrated_risk_gap",
        },
    )

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT obj_error, obj_efficiency, obj_pdm, "
            "window_count, positive_window_count, "
            "pdm_risk_gap, objective_pdm_metric "
            "FROM solutions_finetune_riskgap LIMIT 1"
        ).fetchone()
        assert row is not None
        assert float(row[0]) == 1.25
        assert float(row[1]) == 1234.0
        assert float(row[2]) == 0.33
        assert int(row[3]) == 100
        assert int(row[4]) == 10
        assert float(row[5]) == 0.6
        assert str(row[6]) == "calibrated_risk_gap"
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
        assert "pdm_risk_gap" in columns
        assert "calibration_window_count" in columns
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
        "pdm_signal_quality": 0.6,
        "objective_reason": None,
        "objective_contract": {
            "error_metric": "SMAPE",
            "efficiency_metric": "macs",
            "pdm_metric": "calibrated_risk_gap",
        },
        "metrics": {"SMAPE": 1.25},
        "anomaly_metrics": {
            "pdm_positive_risk_mean": 0.8,
            "pdm_negative_risk_mean": 0.2,
            "pdm_risk_gap": 0.6,
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
    final_training = summary["final_training"]
    assert final_training["training_policy"]["optimizer"] == "Adam"
    assert final_training["obj_error"] == 1.25
    assert final_training["obj_efficiency"] == 1234.0
    assert final_training["obj_pdm"] == 0.33
    assert "error" not in final_training
    assert "complexity" not in final_training
    assert "fitness" not in final_training
    assert "window_auprc" not in final_training["anomaly_metrics"]


def test_shareable_risk_gap_objective_markdown_exists():
    doc_path = Path(__file__).resolve().parents[2] / "CALIBRATED_RISK_GAP_OBJECTIVE.md"
    content = doc_path.read_text(encoding="utf-8")

    assert "pdm_risk_gap" in content
    assert "obj_pdm = clip(0.5 * (1 - pdm_risk_gap), 0, 1)" in content
    assert "risk_gap =  1.0 -> obj_pdm = 0.0" in content


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
