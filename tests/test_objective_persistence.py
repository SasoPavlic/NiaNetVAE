from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from nianetvae.search.runtime_artifacts import _export_cycle_artifacts
from nianetvae.storage.experiment_storage import SQLiteConnector


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.hash_id = "dummy-hash"
        self.activation_name = "Tanh"
        self.optimizer_name = "ASGD"
        self.encoder_layer_step = 16
        self.encoder_num_layers = 2
        self.decoder_num_layers = 2
        self.decoder_layer_step = 16
        self.encoding_layers = [3, 2]
        self.decoding_layers = [2, 3]
        self.bottleneck_size = 2
        self.mapping_context = {"source": "test"}


def test_sqlite_objective_only_schema_and_insert(tmp_path: Path):
    db_path = tmp_path / "objective_only.sqlite"
    connector = SQLiteConnector(str(db_path), "solutions")

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(solutions)").fetchall()}
        assert {"obj_error", "obj_efficiency", "obj_pdm"}.issubset(columns)
        assert "error" not in columns
        assert "complexity" not in columns
        assert "fitness" not in columns
    finally:
        conn.close()

    model = _DummyModel()
    connector.save_model_and_entry(
        dataset_name="MetroPT_cycle00",
        alg_name="NSGA3",
        iteration=1,
        solution=np.asarray([0.1] * 7, dtype=float),
        model=model,
        experiment=None,
        obj_error=1.25,
        obj_efficiency=1234.0,
        obj_pdm=0.33,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT obj_error, obj_efficiency, obj_pdm FROM solutions LIMIT 1"
        ).fetchone()
        assert row is not None
        assert float(row[0]) == 1.25
        assert float(row[1]) == 1234.0
        assert float(row[2]) == 0.33
    finally:
        conn.close()

    df = connector.get_cycle_candidates("MetroPT_cycle00", algorithm_name="NSGA3")
    assert {"obj_error", "obj_efficiency", "obj_pdm"}.issubset(set(df.columns))
    assert "error" not in set(df.columns)
    assert "complexity" not in set(df.columns)
    assert "fitness" not in set(df.columns)


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
        "exp_params": {"manual_seed": 42},
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
            "pdm_metric": "auprc_premaint",
        },
        "metrics": {"SMAPE": 1.25},
        "anomaly_metrics": {"pr_auc_mean": 0.67},
    }

    _, meta_path, summary_path = _export_cycle_artifacts(
        export_dir=export_dir,
        model=model,
        best_solution=np.asarray([0.1] * 7, dtype=float),
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

    final_training = summary["final_training"]
    assert final_training["obj_error"] == 1.25
    assert final_training["obj_efficiency"] == 1234.0
    assert final_training["obj_pdm"] == 0.33
    assert "error" not in final_training
    assert "complexity" not in final_training
    assert "fitness" not in final_training
