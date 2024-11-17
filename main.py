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
from nianetvae.dataloaders.smap_and_msl_dataloader import SMABandMSDataLoader
from nianetvae.dataloaders.smd_dataloader import SMDDataLoader
from nianetvae.dataloaders.swat_dataloader import SWATDataLoader
from nianetvae.dataloaders.ucr_dataloader import UCRDataLoader
from nianetvae.dataloaders.wadi_dataloader import WADIDataLoader
from nianetvae.dataloaders.yahoo_dataloader import YahooA1DataLoader
from nianetvae.storage.database import SQLiteConnector
from nianetvae.rnn_vae_architecture_search import solve_architecture_problem

def select_dataloader(config):
    dataset_name = config["data_params"].get("dataset_name", "")

    # Define a mapping of dataset types to DataLoader classes
    dataloader_switch = {
        "YahooA1": YahooA1DataLoader,
        "KPI": KPIDataLoader,
        "MSL": SMABandMSDataLoader,
        "SMAP": SMABandMSDataLoader,  # Use the same data loader for SMAP & MSL
        "SMD": SMDDataLoader,
        "UCR": UCRDataLoader,
        "SWAT": SWATDataLoader,
        "WADI": WADIDataLoader,
        # Add other datasets as needed
    }

    # Get the appropriate DataLoader class based on the dataset_name
    DataLoaderClass = dataloader_switch.get(dataset_name)

    if DataLoaderClass is None:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

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

    parser.add_argument('--algorithms', '-alg',
                        dest="algorithms",
                        metavar='list_of_strings',
                        help='NIA algorithms to use')

    args = parser.parse_args()

    # Load main configuration
    with open(args.filename, 'r') as file:
        try:
            config = yaml.load(file, Loader=yaml.Loader)  # yaml.safe_load(file)
        except yaml.YAMLError as exc:
            Log.error("Error while loading config file")
            Log.error(exc)
            exit(1)

    # Load dataset-specific configuration
    dataset_config_file = config['dataset']['config_file']
    with open(dataset_config_file, 'r') as file:
        try:
            dataset_config = yaml.load(file, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            Log.error(f"Error while loading dataset config file: {dataset_config_file}")
            Log.error(exc)
            exit(1)

    # Merge dataset_config into config
    config.update(dataset_config)

    # Merge shared data loader parameters into data_params
    shared_data_loader_params = config.get('data_loader_params', {})
    if 'data_params' not in config:
        config['data_params'] = {}
    config['data_params'].update(shared_data_loader_params)

    # Continue with the rest of the code
    config['logging_params']['save_dir'] += RUN_UUID + '/'
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

    conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = select_dataloader(config)
    datamodule.setup()

    nianetvae.rnn_vae_architecture_search.RUN_UUID = RUN_UUID
    nianetvae.rnn_vae_architecture_search.config = config
    nianetvae.rnn_vae_architecture_search.conn = conn
    nianetvae.rnn_vae_architecture_search.datamodule = datamodule

    algorithms = []
    if args.algorithms is not None:
        args.algorithms = args.algorithms.split(',')
        algorithms = args.algorithms
    else:
        algorithms = config['nia_search']['algorithms']

    solve_architecture_problem(algorithms)
    Log.info(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
