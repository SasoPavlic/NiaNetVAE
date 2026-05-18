import pytest

import main


def test_resolve_objective_contract_defaults_when_missing() -> None:
    cfg = {}

    contract = main._resolve_objective_contract(cfg)

    assert contract == {
        "error": {"metric": "SMAPE"},
        "efficiency": {"metric": "macs"},
        "pdm": {"metric": "smoothed_rank_gap", "smoothing_window_windows": 480},
    }
    assert cfg["objectives"] == contract


def test_resolve_objective_contract_normalizes_case() -> None:
    cfg = {
        "objectives": {
            "error": {"metric": "rmse"},
            "efficiency": {"metric": "MACS"},
            "pdm": {"metric": "SMOOTHED_RANK_GAP", "smoothing_window_windows": "240"},
        }
    }

    contract = main._resolve_objective_contract(cfg)

    assert contract["error"]["metric"] == "RMSE"
    assert contract["efficiency"]["metric"] == "macs"
    assert contract["pdm"]["metric"] == "smoothed_rank_gap"
    assert contract["pdm"]["smoothing_window_windows"] == 240


def test_resolve_objective_contract_preserves_selection_block() -> None:
    cfg = {
        "objectives": {
            "error": {"metric": "smape"},
            "selection": {
                "method": "weighted_ideal_distance",
                "weights": {"error": 0.3, "efficiency": 0.2, "pdm": 0.5},
            },
        }
    }

    contract = main._resolve_objective_contract(cfg)

    assert "selection" in contract
    assert contract["selection"]["method"] == "weighted_ideal_distance"


def test_resolve_objective_contract_invalid_error_metric_raises() -> None:
    cfg = {"objectives": {"error": {"metric": "BAD_ERROR"}}}

    with pytest.raises(ValueError, match="objectives.error.metric"):
        main._resolve_objective_contract(cfg)


def test_resolve_objective_contract_invalid_efficiency_metric_raises() -> None:
    cfg = {"objectives": {"efficiency": {"metric": "BAD_EFF"}}}

    with pytest.raises(ValueError, match="objectives.efficiency.metric"):
        main._resolve_objective_contract(cfg)


def test_resolve_objective_contract_invalid_pdm_metric_raises() -> None:
    cfg = {"objectives": {"pdm": {"metric": "BAD_PDM"}}}

    with pytest.raises(ValueError, match="objectives.pdm.metric"):
        main._resolve_objective_contract(cfg)


def test_resolve_objective_contract_rejects_legacy_window_auprc_metric() -> None:
    cfg = {"objectives": {"pdm": {"metric": "window_auprc"}}}
    with pytest.raises(ValueError, match="objectives.pdm.metric"):
        main._resolve_objective_contract(cfg)


@pytest.mark.parametrize("legacy_metric", ["fixed_theta_fbeta_covpen", "calibrated_risk_fbeta_covpen", "calibrated_risk_gap"])
def test_resolve_objective_contract_rejects_old_pdm_metrics(legacy_metric: str) -> None:
    cfg = {"objectives": {"pdm": {"metric": legacy_metric}}}
    with pytest.raises(ValueError, match="objectives.pdm.metric"):
        main._resolve_objective_contract(cfg)


def test_resolve_objective_contract_rejects_legacy_fixed_theta_key() -> None:
    cfg = {
        "objectives": {
            "pdm": {
                "metric": "smoothed_rank_gap",
                "fixed_theta": 0.61,
            }
        }
    }
    with pytest.raises(ValueError, match="fixed_theta"):
        main._resolve_objective_contract(cfg)


@pytest.mark.parametrize(
    "removed_key",
    ["risk_score_exceedance_quantile", "beta", "coverage_target", "coverage_penalty_lambda"],
)
def test_resolve_objective_contract_rejects_removed_pdm_parameters(removed_key: str) -> None:
    cfg = {"objectives": {"pdm": {"metric": "smoothed_rank_gap", removed_key: 0.5}}}
    with pytest.raises(ValueError, match="Removed objectives.pdm keys"):
        main._resolve_objective_contract(cfg)


def test_config_summary_line_includes_objective_contract_fields() -> None:
    cfg = {
        "workflow": {"mode": "per_maint_baseline_search"},
        "data_params": {"dataset_name": "MetroPT"},
        "nia_search": {"metrics": ["SMAPE"], "nsga3": {"n_partitions": 5, "effective_population": 21}},
        "exp_params": {"optimizer": "Adam", "learning_rate": 0.003, "weight_decay": 0.0},
        "objectives": {
            "error": {"metric": "SMAPE"},
            "efficiency": {"metric": "macs"},
            "pdm": {"metric": "smoothed_rank_gap", "smoothing_window_windows": 480},
        },
    }

    line = main._config_summary_line(cfg)

    assert "obj_error=SMAPE" in line
    assert "obj_efficiency=macs" in line
    assert "obj_pdm=smoothed_rank_gap" in line
    assert "nsga3_n_partitions=5" in line
    assert "nsga3_effective_population=21" in line
    assert "fixed_optimizer=Adam" in line
    assert "base_learning_rate=0.003" in line
    assert "weight_decay=0.0" in line


def test_objective_contract_line_contains_locked_semantics() -> None:
    contract = {
        "error": {"metric": "SMAPE"},
        "efficiency": {"metric": "macs"},
        "pdm": {"metric": "smoothed_rank_gap", "smoothing_window_windows": 480},
    }

    line = main._objective_contract_line(contract)

    assert line.startswith("OBJECTIVE_CONTRACT ")
    assert "obj_error=SMAPE direction=min" in line
    assert "obj_efficiency=macs direction=min" in line
    assert "obj_pdm=1-pdm_smoothed_auroc direction=min" in line
    assert "pdm_metric=smoothed_rank_gap" in line
    assert "pdm_smoothing_window_windows=480" in line
    assert "pdm_score_pipeline=window_reconstruction_error->risk_score->smoothed_risk_score->smoothed_rank_gap" in line
    assert "pdm_label_policy=phase0_or_phase1_positive_is_phase1_exclude_phase2" in line
    assert "pdm_eval_slice=test_only" in line


def test_enforce_anomaly_metrics_enabled_forces_exp_param_and_removes_legacy_flag() -> None:
    cfg = {
        "data_params": {"dataset_name": "MetroPT", "compute_anomaly_metrics": False},
        "exp_params": {"compute_anomaly_metrics": False},
    }

    main._enforce_anomaly_metrics_enabled(cfg)

    assert cfg["exp_params"]["compute_anomaly_metrics"] is True
    assert "compute_anomaly_metrics" not in cfg["data_params"]


def test_resolve_training_policy_defaults_and_validation() -> None:
    cfg = {}

    contract = main._resolve_training_policy(cfg)

    assert contract == {"optimizer": "Adam", "learning_rate": 0.003, "weight_decay": 0.0}
    assert cfg["exp_params"]["optimizer"] == "Adam"


def test_resolve_training_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="exp_params.optimizer"):
        main._resolve_training_policy({"exp_params": {"optimizer": "SGD"}})

    with pytest.raises(ValueError, match="exp_params.learning_rate"):
        main._resolve_training_policy({"exp_params": {"optimizer": "Adam", "learning_rate": 0}})

    with pytest.raises(ValueError, match="exp_params.weight_decay"):
        main._resolve_training_policy({"exp_params": {"optimizer": "Adam", "weight_decay": -1}})


def test_validate_pdm_objective_scope_accepts_metropt_per_maint() -> None:
    cfg = {
        "objectives": {"pdm": {"metric": "smoothed_rank_gap"}},
        "data_params": {"dataset_name": "MetroPT", "regime": "per_maint"},
    }

    main._validate_pdm_objective_scope(cfg)


def test_validate_pdm_objective_scope_rejects_other_scope() -> None:
    cfg = {
        "objectives": {"pdm": {"metric": "smoothed_rank_gap"}},
        "data_params": {"dataset_name": "SMAP", "regime": "single"},
    }

    with pytest.raises(ValueError, match="requires"):
        main._validate_pdm_objective_scope(cfg)


def test_resolve_nsga3_search_config_valid() -> None:
    cfg = {"nia_search": {"nsga3": {"n_partitions": 5}}}

    out = main._resolve_nsga3_search_config(cfg)

    assert out["n_partitions"] == 5
    assert out["effective_population"] == 21
    assert cfg["nia_search"]["nsga3"]["effective_population"] == 21


def test_resolve_nsga3_search_config_rejects_legacy_population_size() -> None:
    cfg = {"nia_search": {"population_size": 20, "nsga3": {"n_partitions": 5}}}

    with pytest.raises(ValueError, match="population_size"):
        main._resolve_nsga3_search_config(cfg)


def test_resolve_nsga3_search_config_rejects_missing_or_invalid_partitions() -> None:
    with pytest.raises(ValueError, match="n_partitions"):
        main._resolve_nsga3_search_config({"nia_search": {"nsga3": {}}})

    with pytest.raises(ValueError, match="n_partitions"):
        main._resolve_nsga3_search_config({"nia_search": {"nsga3": {"n_partitions": 0}}})


def test_resolve_winner_selection_contract_defaults() -> None:
    cfg = {}

    contract = main._resolve_winner_selection_contract(cfg)

    assert contract["method"] == "weighted_ideal_distance"
    assert contract["weights"] == {"error": 0.30, "efficiency": 0.20, "pdm": 0.50}
    assert contract["weights_normalized"]["error"] == pytest.approx(0.30)
    assert contract["weights_normalized"]["efficiency"] == pytest.approx(0.20)
    assert contract["weights_normalized"]["pdm"] == pytest.approx(0.50)
    assert cfg["objectives"]["selection"]["method"] == "weighted_ideal_distance"


def test_resolve_winner_selection_contract_invalid_method_raises() -> None:
    cfg = {"objectives": {"selection": {"method": "bad_method"}}}

    with pytest.raises(ValueError, match="objectives.selection.method"):
        main._resolve_winner_selection_contract(cfg)


def test_resolve_winner_selection_contract_invalid_weights_raise() -> None:
    cfg_negative = {"objectives": {"selection": {"weights": {"error": -1.0}}}}
    with pytest.raises(ValueError, match="weights.error"):
        main._resolve_winner_selection_contract(cfg_negative)

    cfg_zero_sum = {
        "objectives": {
            "selection": {
                "weights": {"error": 0.0, "efficiency": 0.0, "pdm": 0.0}
            }
        }
    }
    with pytest.raises(ValueError, match="sum\\(error, efficiency, pdm\\) must be > 0"):
        main._resolve_winner_selection_contract(cfg_zero_sum)


def test_resolve_db_table_name_defaults_and_normalizes() -> None:
    cfg = {"logging_params": {}}
    table_name = main._resolve_db_table_name(cfg)
    assert table_name == "solutions_finetune_smoothed_rankgap"
    assert cfg["logging_params"]["db_table_name"] == "solutions_finetune_smoothed_rankgap"

    cfg_custom = {"logging_params": {"db_table_name": "  my_table  "}}
    table_name_custom = main._resolve_db_table_name(cfg_custom)
    assert table_name_custom == "my_table"
    assert cfg_custom["logging_params"]["db_table_name"] == "my_table"


def test_resolve_db_table_name_rejects_blank() -> None:
    cfg = {"logging_params": {"db_table_name": "   "}}
    with pytest.raises(ValueError, match="logging_params.db_table_name"):
        main._resolve_db_table_name(cfg)
