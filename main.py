import argparse
import math
import uuid
from datetime import datetime
from pathlib import Path
import torch
import yaml
from lightning.pytorch import seed_everything

from log import Log
from nianetvae.dataloaders.kpi_dataloader import KPIDataLoader
from nianetvae.dataloaders.nab_dataloader import NABDataLoader
from nianetvae.dataloaders.smap_and_msl_dataloader import SMABandMSDataLoader
from nianetvae.dataloaders.smd_dataloader import SMDDataLoader
from nianetvae.dataloaders.swat_dataloader import SWATDataLoader
from nianetvae.dataloaders.ucr_dataloader import UCRDataLoader
from nianetvae.dataloaders.wadi_dataloader import WADIDataLoader
from nianetvae.dataloaders.yahoo_dataloader import YahooA1DataLoader
from nianetvae.dataloaders.metropt_dataloader import MetroPTDataLoader
from nianetvae.search.runner import SearchRunner, SearchRuntimeContext
from nianetvae.storage.experiment_storage import get_db_connector
import nianetvae.experiments.metrics_evaluation

ALLOWED_WORKFLOW_MODES = {"baseline_search", "per_maint_finetune", "per_maint_warmstart_search"}
ALLOWED_ERROR_OBJECTIVE_METRICS = {"MAE", "MSE", "RMSE", "MAPE", "RMAPE", "SMAPE"}
ALLOWED_EFFICIENCY_OBJECTIVE_METRICS = {"params", "macs", "latency_ms"}
ALLOWED_PDM_OBJECTIVE_METRIC = "calibrated_risk_gap"
DEFAULT_DB_TABLE_NAME = "solutions_finetune_riskgap"
ALLOWED_SELECTION_METHODS = {"weighted_ideal_distance"}
ALLOWED_FIXED_OPTIMIZERS = {"Adam"}


def _load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        try:
            data = yaml.load(file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            # main.py loads configs before Log.enable(); do not call Log here.
            raise ValueError(f"Error while loading config file: {path}: {exc}") from exc
    return data or {}


def _err(msg: str) -> None:
    # main.py can error before Log.enable(); keep this stderr-only and dependency-free.
    import sys

    sys.stderr.write(str(msg).rstrip() + "\n")


def select_dataloader(config):
    dataset_name = config["data_params"].get("dataset_name", "")
    dataset_key = str(dataset_name).strip()
    dataset_key_l = dataset_key.lower()

    # Define a mapping of dataset types to DataLoader classes
    dataloader_switch = {
        "yahooa1": YahooA1DataLoader,
        "kpi": KPIDataLoader,
        "msl": SMABandMSDataLoader,
        "smap": SMABandMSDataLoader,  # Use the same data loader for SMAP & MSL
        "smd": SMDDataLoader,
        "ucr": UCRDataLoader,
        "swat": SWATDataLoader,
        "wadi": WADIDataLoader,
        "nab": NABDataLoader,
        "metropt": MetroPTDataLoader,
        # Add other datasets as needed
    }

    # Get the appropriate DataLoader class based on the dataset_name
    DataLoaderClass = dataloader_switch.get(dataset_key_l)

    if DataLoaderClass is None:
        raise ValueError(
            f"Unsupported dataset name: {dataset_name!r}. "
            f"Expected one of: {sorted(dataloader_switch.keys())}"
        )

    # Initialize the DataLoader with the corresponding parameters
    return DataLoaderClass(**config["data_params"])


def _as_csv(values) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    try:
        return ",".join(str(v) for v in values)
    except Exception:
        return str(values)


def _value_or_na(value):
    if value is None:
        return "n/a"
    if isinstance(value, str) and not value.strip():
        return "n/a"
    return value


def _config_summary_line(config: dict) -> str:
    data_params = config.get("data_params", {})
    nia_search = config.get("nia_search", {})
    nsga3_cfg = nia_search.get("nsga3") or {}
    logging_params = config.get("logging_params", {})
    trainer_params = config.get("trainer_params", {})
    exp_params = config.get("exp_params", {})
    workflow_mode = ((config.get("workflow") or {}).get("mode"))
    parts = [
        "CONFIG_SUMMARY",
        f"workflow_mode={_value_or_na(workflow_mode)}",
        f"dataset={_value_or_na(data_params.get('dataset_name'))}",
        f"regime={_value_or_na(data_params.get('regime'))}",
        f"cycle_id={_value_or_na(data_params.get('cycle_id'))}",
        f"seq_len={_value_or_na(data_params.get('seq_len'))}",
        f"batch_size={_value_or_na(data_params.get('batch_size'))}",
        f"n_features={_value_or_na(data_params.get('n_features'))}",
        f"min_epochs={_value_or_na(trainer_params.get('min_epochs'))}",
        f"max_epochs={_value_or_na(trainer_params.get('max_epochs'))}",
        f"nsga3_n_partitions={_value_or_na(nsga3_cfg.get('n_partitions'))}",
        f"nsga3_effective_population={_value_or_na(nsga3_cfg.get('effective_population'))}",
        f"time_limit={_value_or_na(nia_search.get('time'))}",
        f"metrics={_as_csv(nia_search.get('metrics'))}",
        f"obj_error={_value_or_na(((config.get('objectives') or {}).get('error') or {}).get('metric'))}",
        f"obj_efficiency={_value_or_na(((config.get('objectives') or {}).get('efficiency') or {}).get('metric'))}",
        f"obj_pdm={_value_or_na(((config.get('objectives') or {}).get('pdm') or {}).get('metric'))}",
        f"winner_selection={_value_or_na((((config.get('objectives') or {}).get('selection') or {}).get('method')))}",
        "winner_weights="
        f"{_value_or_na(((((config.get('objectives') or {}).get('selection') or {}).get('weights') or {}).get('error')))}"
        "/"
        f"{_value_or_na(((((config.get('objectives') or {}).get('selection') or {}).get('weights') or {}).get('efficiency')))}"
        "/"
        f"{_value_or_na(((((config.get('objectives') or {}).get('selection') or {}).get('weights') or {}).get('pdm')))}",
        f"fixed_optimizer={_value_or_na(exp_params.get('optimizer'))}",
        f"base_learning_rate={_value_or_na(exp_params.get('learning_rate'))}",
        f"weight_decay={_value_or_na(exp_params.get('weight_decay'))}",
        f"db_backend={_value_or_na(logging_params.get('db_backend', 'sqlite'))}",
        f"db_table={_value_or_na(logging_params.get('db_table_name', DEFAULT_DB_TABLE_NAME))}",
    ]
    return " ".join(parts)


def _resolve_workflow_mode(config: dict) -> str:
    workflow = config.get("workflow") or {}
    raw_mode = workflow.get("mode", "baseline_search")
    mode = str(raw_mode).strip().lower() if raw_mode is not None else "baseline_search"
    if not mode:
        mode = "baseline_search"
    if mode not in ALLOWED_WORKFLOW_MODES:
        allowed = ", ".join(sorted(ALLOWED_WORKFLOW_MODES))
        raise ValueError(
            f"Invalid workflow.mode={raw_mode!r}. Allowed values: {allowed}."
        )
    normalized_workflow = dict(workflow)
    normalized_workflow["mode"] = mode
    config["workflow"] = normalized_workflow
    return mode


def _resolve_nsga3_search_config(config: dict) -> dict:
    """
    C13.4 contract lock:
    - NSGA-III only (Das-Dennis reference directions)
    - no legacy nia_search.population_size
    - required nia_search.nsga3.n_partitions >= 1
    """
    nia_search = dict(config.get("nia_search") or {})
    if "population_size" in nia_search:
        raise ValueError(
            "Legacy config key nia_search.population_size is no longer supported in C13.4. "
            "Use nia_search.nsga3.n_partitions for NSGA-III sizing."
        )

    nsga3_cfg = dict(nia_search.get("nsga3") or {})
    if "n_partitions" not in nsga3_cfg:
        raise ValueError(
            "Missing required config key nia_search.nsga3.n_partitions for NSGA-III."
        )
    raw_partitions = nsga3_cfg.get("n_partitions")
    try:
        n_partitions = int(raw_partitions)
    except Exception:
        raise ValueError(
            f"Invalid nia_search.nsga3.n_partitions={raw_partitions!r}. Expected integer >= 1."
        ) from None
    if n_partitions < 1:
        raise ValueError(
            f"Invalid nia_search.nsga3.n_partitions={raw_partitions!r}. Expected integer >= 1."
        )

    # For 3 objectives, Das-Dennis count is C(p+2, 2) = (p+2)*(p+1)/2.
    effective_population = int((n_partitions + 2) * (n_partitions + 1) // 2)
    normalized_nsga3_cfg = {
        "n_partitions": n_partitions,
        "effective_population": effective_population,
    }
    nia_search["nsga3"] = normalized_nsga3_cfg
    config["nia_search"] = nia_search
    return normalized_nsga3_cfg


def _resolve_objective_contract(config: dict) -> dict:
    """
    C13.1 contract lock:
      - obj_error: single selected reconstruction metric (min)
      - obj_efficiency: selected efficiency backend (min)
      - obj_pdm: clip(0.5 * (1 - calibrated_risk_gap), 0, 1) (min)
    """
    objectives = dict(config.get("objectives") or {})
    error_cfg = dict(objectives.get("error") or {})
    efficiency_cfg = dict(objectives.get("efficiency") or {})
    pdm_cfg = dict(objectives.get("pdm") or {})

    raw_error_metric = error_cfg.get("metric", "SMAPE")
    error_metric = str(raw_error_metric).strip().upper() if raw_error_metric is not None else "SMAPE"
    if error_metric not in ALLOWED_ERROR_OBJECTIVE_METRICS:
        allowed = ", ".join(sorted(ALLOWED_ERROR_OBJECTIVE_METRICS))
        raise ValueError(
            f"Invalid objectives.error.metric={raw_error_metric!r}. Allowed values: {allowed}."
        )

    raw_efficiency_metric = efficiency_cfg.get("metric", "macs")
    efficiency_metric = (
        str(raw_efficiency_metric).strip().lower() if raw_efficiency_metric is not None else "macs"
    )
    if efficiency_metric not in ALLOWED_EFFICIENCY_OBJECTIVE_METRICS:
        allowed = ", ".join(sorted(ALLOWED_EFFICIENCY_OBJECTIVE_METRICS))
        raise ValueError(
            f"Invalid objectives.efficiency.metric={raw_efficiency_metric!r}. "
            f"Allowed values: {allowed}."
        )

    raw_pdm_metric = pdm_cfg.get("metric", ALLOWED_PDM_OBJECTIVE_METRIC)
    pdm_metric = str(raw_pdm_metric).strip().lower() if raw_pdm_metric is not None else ALLOWED_PDM_OBJECTIVE_METRIC
    if pdm_metric != ALLOWED_PDM_OBJECTIVE_METRIC:
        raise ValueError(
            f"Invalid objectives.pdm.metric={raw_pdm_metric!r}. "
            f"Allowed value: {ALLOWED_PDM_OBJECTIVE_METRIC}."
        )
    if "fixed_theta" in pdm_cfg:
        raise ValueError(
            "Legacy objectives.pdm.fixed_theta is no longer supported for NiaNetVAE. "
            "Downstream maintenance_risk_theta belongs to metropt-pdm-framework evaluation only."
        )
    removed_keys = {
        "risk_score_exceedance_quantile",
        "beta",
        "coverage_target",
        "coverage_penalty_lambda",
    }
    present_removed_keys = sorted(key for key in removed_keys if key in pdm_cfg)
    if present_removed_keys:
        raise ValueError(
            "Removed objectives.pdm keys are no longer supported for NiaNetVAE "
            f"calibrated_risk_gap objective: {present_removed_keys}. "
            "Coverage/recall targets belong to metropt-pdm-framework evaluation."
        )

    normalized_contract = {
        "error": {"metric": error_metric},
        "efficiency": {"metric": efficiency_metric},
        "pdm": {
            "metric": pdm_metric,
        },
    }
    if "selection" in objectives:
        normalized_contract["selection"] = dict(objectives.get("selection") or {})
    config["objectives"] = normalized_contract
    return normalized_contract


def _resolve_winner_selection_contract(config: dict) -> dict:
    objectives = dict(config.get("objectives") or {})
    selection_cfg = dict(objectives.get("selection") or {})
    raw_method = selection_cfg.get("method", "weighted_ideal_distance")
    method = str(raw_method).strip().lower() if raw_method is not None else "weighted_ideal_distance"
    if method not in ALLOWED_SELECTION_METHODS:
        allowed = ", ".join(sorted(ALLOWED_SELECTION_METHODS))
        raise ValueError(
            f"Invalid objectives.selection.method={raw_method!r}. Allowed values: {allowed}."
        )

    raw_weights_cfg = dict(selection_cfg.get("weights") or {})
    defaults = {"error": 0.30, "efficiency": 0.20, "pdm": 0.50}
    resolved_weights = {}
    for key in ("error", "efficiency", "pdm"):
        raw_value = raw_weights_cfg.get(key, defaults[key])
        try:
            weight = float(raw_value)
        except Exception:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            ) from None
        if not math.isfinite(weight):
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            )
        if weight < 0:
            raise ValueError(
                f"Invalid objectives.selection.weights.{key}={raw_value!r}. Expected finite float >= 0."
            )
        resolved_weights[key] = weight

    total = resolved_weights["error"] + resolved_weights["efficiency"] + resolved_weights["pdm"]
    if total <= 0:
        raise ValueError(
            "Invalid objectives.selection.weights: sum(error, efficiency, pdm) must be > 0."
        )
    normalized_weights = {
        "error": resolved_weights["error"] / total,
        "efficiency": resolved_weights["efficiency"] / total,
        "pdm": resolved_weights["pdm"] / total,
    }

    normalized_selection = {
        "method": method,
        "weights": resolved_weights,
        "weights_normalized": normalized_weights,
    }
    objectives["selection"] = {
        "method": method,
        "weights": resolved_weights,
    }
    config["objectives"] = objectives
    return normalized_selection


def _resolve_training_policy(config: dict) -> dict:
    exp_params = dict(config.get("exp_params") or {})

    raw_optimizer = exp_params.get("optimizer", "Adam")
    optimizer_name = str(raw_optimizer).strip() if raw_optimizer is not None else "Adam"
    if optimizer_name not in ALLOWED_FIXED_OPTIMIZERS:
        allowed = ", ".join(sorted(ALLOWED_FIXED_OPTIMIZERS))
        raise ValueError(
            f"Invalid exp_params.optimizer={raw_optimizer!r}. Allowed values: {allowed}."
        )

    raw_lr = exp_params.get("learning_rate", 0.003)
    try:
        learning_rate = float(raw_lr)
    except Exception:
        raise ValueError(
            f"Invalid exp_params.learning_rate={raw_lr!r}. Expected finite float > 0."
        ) from None
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            f"Invalid exp_params.learning_rate={raw_lr!r}. Expected finite float > 0."
        )

    raw_weight_decay = exp_params.get("weight_decay", 0.0)
    try:
        weight_decay = float(raw_weight_decay)
    except Exception:
        raise ValueError(
            f"Invalid exp_params.weight_decay={raw_weight_decay!r}. Expected finite float >= 0."
        ) from None
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            f"Invalid exp_params.weight_decay={raw_weight_decay!r}. Expected finite float >= 0."
        )

    exp_params["optimizer"] = optimizer_name
    exp_params["learning_rate"] = learning_rate
    exp_params["weight_decay"] = weight_decay
    config["exp_params"] = exp_params
    return {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def _objective_contract_line(contract: dict) -> str:
    error_metric = ((contract.get("error") or {}).get("metric"))
    efficiency_metric = ((contract.get("efficiency") or {}).get("metric"))
    pdm_cfg = contract.get("pdm") or {}
    pdm_metric = pdm_cfg.get("metric")
    return (
        "OBJECTIVE_CONTRACT "
        f"obj_error={_value_or_na(error_metric)} direction=min "
        f"obj_efficiency={_value_or_na(efficiency_metric)} direction=min "
        "obj_pdm=clip(0.5*(1-pdm_risk_gap),0,1) direction=min "
        f"pdm_metric={_value_or_na(pdm_metric)} "
        "pdm_score_pipeline=window_reconstruction_error->risk_score->risk_gap "
        "pdm_label_policy=phase0_or_phase1_positive_is_phase1_exclude_phase2 "
        "pdm_eval_slice=test_only"
    )


def _winner_selection_contract_line(contract: dict) -> str:
    method = contract.get("method")
    weights = contract.get("weights_normalized") or {}
    error_w = float(weights.get("error", 0.0))
    efficiency_w = float(weights.get("efficiency", 0.0))
    pdm_w = float(weights.get("pdm", 0.0))
    return (
        "WINNER_SELECTION_CONTRACT "
        f"method={_value_or_na(method)} "
        f"weights_normalized="
        f"{error_w:.4f}/"
        f"{efficiency_w:.4f}/"
        f"{pdm_w:.4f} "
        "pool=db_cycle_dataset_name pareto_front_only normalization=minmax_per_pool "
        "tie_break=obj_pdm_then_obj_error_then_obj_efficiency_then_timestamp_then_id "
        "empty_pool_policy=fail_fast"
    )


def _training_policy_contract_line(contract: dict) -> str:
    return (
        "TRAINING_POLICY_CONTRACT "
        f"optimizer={_value_or_na(contract.get('optimizer'))} "
        f"learning_rate={_value_or_na(contract.get('learning_rate'))} "
        f"weight_decay={_value_or_na(contract.get('weight_decay'))} "
        "search_space=architecture_only"
    )


def _enforce_anomaly_metrics_enabled(config: dict) -> None:
    """
    C13.3 policy lock:
    anomaly metrics are always enabled for search/runtime objective computation.
    """
    data_params = config.get("data_params") or {}
    if "compute_anomaly_metrics" in data_params:
        # Legacy compatibility: remove old dataset-level toggle to avoid conflicting semantics.
        data_params = dict(data_params)
        data_params.pop("compute_anomaly_metrics", None)
        config["data_params"] = data_params

    exp_params = dict(config.get("exp_params") or {})
    exp_params["compute_anomaly_metrics"] = True
    config["exp_params"] = exp_params


def _validate_pdm_objective_scope(config: dict) -> None:
    pdm_metric = str((((config.get("objectives") or {}).get("pdm") or {}).get("metric") or "").strip()).lower()
    if pdm_metric != ALLOWED_PDM_OBJECTIVE_METRIC:
        return

    data_params = config.get("data_params") or {}
    dataset_name = str(data_params.get("dataset_name") or "").strip().lower()
    regime = str(data_params.get("regime") or "").strip().lower()
    if dataset_name != "metropt" or regime != "per_maint":
        raise ValueError(
            f"objectives.pdm.metric='{ALLOWED_PDM_OBJECTIVE_METRIC}' requires "
            "data_params.dataset_name='MetroPT' and data_params.regime='per_maint'."
        )


def _resolve_db_table_name(config: dict) -> str:
    logging_params = dict(config.get("logging_params") or {})
    raw_name = logging_params.get("db_table_name", DEFAULT_DB_TABLE_NAME)
    table_name = str(raw_name).strip() if raw_name is not None else DEFAULT_DB_TABLE_NAME
    if not table_name:
        raise ValueError(
            f"Invalid logging_params.db_table_name={raw_name!r}. Expected non-empty string."
        )
    logging_params["db_table_name"] = table_name
    config["logging_params"] = logging_params
    return table_name


def _is_non_trainable_cycle_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "zero rows after phase filtering" in message
        or "zero positive windows after phase filtering" in message
    )


if __name__ == '__main__':

    RUN_UUID = uuid.uuid4().hex
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/main_config.yaml')

    parser.add_argument('--metrics', '-met',
                        dest="metrics",
                        metavar='list_of_strings',
                        help='Metrics to calculate (comma-separated)')
    parser.add_argument('--cycle-id',
                        dest="cycle_id",
                        type=int,
                        help='MetroPT cycle id for per_maint (0=pre_W1, 1..21=maintenance windows). When set, regime is forced to per_maint.')

    args = parser.parse_args()

    # Load main configuration
    try:
        config = _load_yaml_file(args.filename)
    except Exception:
        _err(f"Failed to load config: {args.filename}")
        exit(1)

    # Load dataset-specific configuration. Support chained configs where the loaded dataset config
    # points to another dataset.config_file (common "entrypoint wrapper" pattern).
    seen = set()
    for _ in range(5):  # prevent infinite loops
        dataset_cfg_path = (config.get("dataset") or {}).get("config_file")
        if not dataset_cfg_path:
            break

        import os

        norm_path = os.path.normpath(str(dataset_cfg_path))
        if norm_path in seen:
            _err(f"Detected recursive dataset config_file reference: {dataset_cfg_path}")
            exit(1)
        seen.add(norm_path)
        try:
            dataset_config = _load_yaml_file(dataset_cfg_path)
        except Exception:
            _err(f"Failed to load dataset config: {dataset_cfg_path}")
            exit(1)
        config.update(dataset_config)

        # Only continue chaining if the loaded dataset config itself points to another config_file.
        next_cfg = (dataset_config.get("dataset") or {}).get("config_file")
        if not next_cfg:
            # Prevent re-loading the same dataset config on the next loop iteration.
            if isinstance(config.get("dataset"), dict):
                config["dataset"].pop("config_file", None)
            break

    # Merge shared data loader parameters into data_params
    shared_data_loader_params = config.get('data_loader_params', {})
    if 'data_params' not in config:
        config['data_params'] = {}
    config['data_params'].update(shared_data_loader_params)

    # CLI override for MetroPT cycle: force per_maint and set cycle_id.
    if args.cycle_id is not None:
        config['data_params']['cycle_id'] = int(args.cycle_id)
        config['data_params']['regime'] = "per_maint"

    _enforce_anomaly_metrics_enabled(config)

    try:
        workflow_mode = _resolve_workflow_mode(config)
        _resolve_nsga3_search_config(config)
        objective_contract = _resolve_objective_contract(config)
        training_policy_contract = _resolve_training_policy(config)
        winner_selection_contract = _resolve_winner_selection_contract(config)
    except ValueError as exc:
        _err(str(exc))
        exit(1)

    # Validate that the dataset config provided the required keys before running anything expensive.
    if not config.get("data_params", {}).get("dataset_name"):
        _err(
            "Missing required config key: data_params.dataset_name. "
            "Ensure your dataset config file defines data_params.dataset_name (e.g., 'MetroPT')."
        )
        exit(1)

    # DB dataset naming for per-cycle isolation (single shared SQLite).
    base_dataset_name = str(config["data_params"]["dataset_name"])
    regime = str(config["data_params"].get("regime", "")).strip().lower()
    cycle_id = config["data_params"].get("cycle_id")
    db_dataset_name = base_dataset_name
    if regime == "per_maint" and cycle_id is not None:
        try:
            cid = int(cycle_id)
            db_dataset_name = f"{base_dataset_name}_cycle{cid:02d}"
        except Exception:
            db_dataset_name = base_dataset_name

    try:
        _validate_pdm_objective_scope(config)
    except ValueError as exc:
        _err(str(exc))
        exit(1)

    if workflow_mode in {"per_maint_finetune", "per_maint_warmstart_search"}:
        if regime != "per_maint":
            _err(
                f"workflow.mode='{workflow_mode}' requires "
                "data_params.regime='per_maint'."
            )
            exit(1)
        if cycle_id is None:
            _err(
                f"workflow.mode='{workflow_mode}' requires "
                "data_params.cycle_id to be set."
            )
            exit(1)

    # Continue with the rest of the code
    config['logging_params']['save_dir'] += '/' + RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

    Log.enable(config['logging_params'])
    regime_tag = regime or "n/a"
    cycle_tag = cycle_id if cycle_id is not None else "n/a"
    Log.info(
        f"RUN_START run_uuid={RUN_UUID} dataset={base_dataset_name} "
        f"db_dataset={db_dataset_name} workflow_mode={workflow_mode} "
        f"regime={regime_tag} cycle_id={cycle_tag} "
        f"started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    cuda_available = torch.cuda.is_available()
    Log.info(
        f"ENV torch={torch.__version__} cuda_compiled={torch.version.cuda} "
        f"cuda_available={str(cuda_available).lower()}"
    )
    Log.info(_config_summary_line(config))
    Log.info(_objective_contract_line(objective_contract))
    Log.info(_training_policy_contract_line(training_policy_contract))
    Log.info(_winner_selection_contract_line(winner_selection_contract))

    db_table_name = _resolve_db_table_name(config)
    Log.info(f"DB_TABLE_ROUTING table_name={db_table_name}")
    conn = get_db_connector(config, db_table_name)
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = select_dataloader(config)
    runtime_ctx = SearchRuntimeContext(
        run_uuid=RUN_UUID,
        config=config,
        conn=conn,
        datamodule=datamodule,
        dataset_name=db_dataset_name,
    )
    runner = SearchRunner(runtime_ctx)
    try:
        datamodule.setup()
    except Exception as exc:
        finetune_cycle_id = None
        try:
            finetune_cycle_id = int(cycle_id) if cycle_id is not None else None
        except Exception:
            finetune_cycle_id = None
        should_skip_non_trainable = (
            workflow_mode in {"per_maint_finetune", "per_maint_warmstart_search"}
            and regime == "per_maint"
            and finetune_cycle_id is not None
            and finetune_cycle_id > 0
            and _is_non_trainable_cycle_error(exc)
        )
        if should_skip_non_trainable:
            detail = str(exc).strip()
            Log.warning(
                f"PER_MAINT_SKIP mode={workflow_mode} cycle_id={finetune_cycle_id:02d} "
                f"reason=non_trainable_cycle detail={detail}"
            )
            runner.export_skipped_non_trainable_cycle(
                reason="non_trainable_cycle",
                detail=detail,
                source="datamodule.setup",
            )
            Log.info(
                f"RUN_END run_uuid={RUN_UUID} ended_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                "status=skipped_non_trainable"
            )
            exit(0)
        raise

    # Allow dataloaders to override inferred feature dimensionality (e.g., rolling-feature datasets).
    if hasattr(datamodule, "n_features") and getattr(datamodule, "n_features"):
        config["data_params"]["n_features"] = int(getattr(datamodule, "n_features"))

    if workflow_mode == "baseline_search":
        metrics = args.metrics if args.metrics else config['nia_search']['metrics']
        config['nia_search']['metrics'] = metrics
        runner.solve_architecture_problem()
    elif workflow_mode == "per_maint_finetune":
        runner.run_per_maint_finetune_cycle()
    elif workflow_mode == "per_maint_warmstart_search":
        metrics = args.metrics if args.metrics else config['nia_search']['metrics']
        config['nia_search']['metrics'] = metrics
        runner.run_per_maint_warmstart_cycle()

    Log.info(f"RUN_END run_uuid={RUN_UUID} ended_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
