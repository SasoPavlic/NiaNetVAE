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
from nianetvae.rnn_vae_architecture_search import solve_architecture_problem
import nianetvae.experiments.metrics_evaluation


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


if __name__ == '__main__':

    RUN_UUID = uuid.uuid4().hex
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/main_config.yaml')

    # TODO Can be deleted after double checking
    parser.add_argument('--algorithms', '-alg',
                        dest="algorithms",
                        metavar='list_of_strings',
                        help='NIA algorithms to use (comma-separated)')

    parser.add_argument('--metrics', '-met',
                        dest="metrics",
                        metavar='list_of_strings',
                        help='Metrics to calculate (comma-separated)')
    parser.add_argument('--cycle-id',
                        dest="cycle_id",
                        type=int,
                        help='MetroPT cycle id (1-based). When set, regime is forced to per_maint.')

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

    # Continue with the rest of the code
    config['logging_params']['save_dir'] += '/' + RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

    Log.enable(config['logging_params'])
    Log.info(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    Log.info(f"RUN UUID: {RUN_UUID}")
    Log.info(f"PyTorch was compiled with CUDA version: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    Log.info(f"Is CUDA available on this system? {'Yes' if cuda_available else 'No'}")
    Log.info(f"PyTorch version: {torch.__version__}")

    Log.header("NiaNetVAE settings")
    Log.info(config)
    if db_dataset_name != base_dataset_name:
        Log.info(f"DB dataset name: {db_dataset_name} (base dataset: {base_dataset_name})")

    conn = get_db_connector(config, "solutions")
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = select_dataloader(config)
    datamodule.setup()

    # Allow dataloaders to override inferred feature dimensionality (e.g., rolling-feature datasets).
    if hasattr(datamodule, "n_features") and getattr(datamodule, "n_features"):
        config["data_params"]["n_features"] = int(getattr(datamodule, "n_features"))

    nianetvae.rnn_vae_architecture_search.RUN_UUID = RUN_UUID
    nianetvae.rnn_vae_architecture_search.config = config
    nianetvae.rnn_vae_architecture_search.conn = conn
    nianetvae.rnn_vae_architecture_search.datamodule = datamodule
    nianetvae.rnn_vae_architecture_search.dataset_name = db_dataset_name

    # TODO Can be deleted after double checking
    algorithms = []
    if args.algorithms is not None:
        args.algorithms = args.algorithms.split(',')
        algorithms = args.algorithms
    else:
        algorithms = config['nia_search']['algorithms']

    # Update algorithms and metrics based on arguments or config
    algorithms = args.algorithms if args.algorithms else config['nia_search']['algorithms']
    metrics = args.metrics if args.metrics else config['nia_search']['metrics']

    config['nia_search']['metrics'] = metrics
    solve_architecture_problem(algorithms)

    Log.info(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
