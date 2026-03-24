import argparse
import uuid
from datetime import datetime
from pathlib import Path
import torch
import yaml
from lightning.pytorch import seed_everything

import nianetvae
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
from nianetvae.storage.experiment_storage import get_db_connector
from nianetvae.rnn_vae_architecture_search import solve_architecture_problem, run_per_maint_finetune_cycle
import nianetvae.experiments.metrics_evaluation

ALLOWED_WORKFLOW_MODES = {"baseline_search", "per_maint_finetune"}


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
    logging_params = config.get("logging_params", {})
    trainer_params = config.get("trainer_params", {})
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
        f"population_size={_value_or_na(nia_search.get('population_size'))}",
        f"time_limit={_value_or_na(nia_search.get('time'))}",
        f"metrics={_as_csv(nia_search.get('metrics'))}",
        f"db_backend={_value_or_na(logging_params.get('db_backend', 'sqlite'))}",
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


def _is_non_trainable_cycle_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "zero rows after phase filtering" in message


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

    # Optional: allow dataset configs to control anomaly-metrics computation without overriding exp_params.
    # This keeps dataset configs "data_params-only" while still enabling MetroPT-style runs that do not
    # have ground-truth anomaly labels.
    if "compute_anomaly_metrics" in config.get("data_params", {}):
        config.setdefault("exp_params", {})
        compute_flag = config["data_params"].get("compute_anomaly_metrics")
        config["exp_params"]["compute_anomaly_metrics"] = bool(compute_flag)
        # Avoid passing this key into dataloaders (even though most accept **kwargs).
        config["data_params"].pop("compute_anomaly_metrics", None)

    try:
        workflow_mode = _resolve_workflow_mode(config)
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

    if workflow_mode == "per_maint_finetune":
        if regime != "per_maint":
            _err(
                "workflow.mode='per_maint_finetune' requires "
                "data_params.regime='per_maint'."
            )
            exit(1)
        if cycle_id is None:
            _err(
                "workflow.mode='per_maint_finetune' requires "
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

    conn = get_db_connector(config, "solutions")
    seed_everything(config['exp_params']['manual_seed'], True)
    nianetvae.rnn_vae_architecture_search.RUN_UUID = RUN_UUID
    nianetvae.rnn_vae_architecture_search.config = config
    nianetvae.rnn_vae_architecture_search.conn = conn
    nianetvae.rnn_vae_architecture_search.dataset_name = db_dataset_name

    datamodule = select_dataloader(config)
    try:
        datamodule.setup()
    except Exception as exc:
        finetune_cycle_id = None
        try:
            finetune_cycle_id = int(cycle_id) if cycle_id is not None else None
        except Exception:
            finetune_cycle_id = None
        should_skip_non_trainable = (
            workflow_mode == "per_maint_finetune"
            and regime == "per_maint"
            and finetune_cycle_id is not None
            and finetune_cycle_id > 0
            and _is_non_trainable_cycle_error(exc)
        )
        if should_skip_non_trainable:
            detail = str(exc).strip()
            Log.warning(
                f"FINETUNE_SKIP cycle_id={finetune_cycle_id:02d} "
                f"reason=non_trainable_cycle detail={detail}"
            )
            nianetvae.rnn_vae_architecture_search.export_skipped_non_trainable_cycle(
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

    nianetvae.rnn_vae_architecture_search.datamodule = datamodule

    if workflow_mode == "baseline_search":
        metrics = args.metrics if args.metrics else config['nia_search']['metrics']
        config['nia_search']['metrics'] = metrics
        solve_architecture_problem()
    elif workflow_mode == "per_maint_finetune":
        run_per_maint_finetune_cycle()

    Log.info(f"RUN_END run_uuid={RUN_UUID} ended_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
