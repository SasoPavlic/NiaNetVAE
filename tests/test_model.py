import argparse
import os
import statistics
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error

from log import Log
from nianetvae.dataloaders import TimeSeriesDataset
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment
from nianetvae.models import *

RUN_UUID = uuid.uuid4().hex
parser = argparse.ArgumentParser(description='Generic runner for LSTM VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/main_config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        Log.error(exc)

config['logging_params']['save_dir'] += RUN_UUID + '/'
Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


def fittest_model(existing_model, **kwargs):
    datamodule = TimeSeriesDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    datamodule.setup()

    if existing_model:
        model = torch.load(kwargs["model_path"])
    else:
        model = vae_models[config['model_params']['name']](kwargs["solution"], **config)
        experiment = RNNVAExperiment(model, config['exp_params'], config['model_params']['n_features'])
        config['trainer_params']['max_epochs'] = model.num_epochs
        tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'] + 'all_models/',
                                      name=str("Manual" + "_" + model.hash_id))

        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True),
                             early_stop_callback,
                         ],
                         # strategy=DDPPlugin(find_unused_parameters=False),

                         **config['trainer_params'])

        Log.debug(f"======= Training {config['model_params']['name']} =======")

        Log.debug(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
        runner.fit(experiment, datamodule=datamodule)
        Log.debug(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

    dataloader_iterator = iter(datamodule.test_dataloader())
    list_RMSE = list()

    while True:
        try:
            data, target = next(dataloader_iterator)
        except StopIteration:
            break
        finally:
            predictions = model(data)[0].detach().numpy()
            RMSE = mean_squared_error(data, predictions, squared=False)
            list_RMSE.append(RMSE)

    Log.info(f"Mean RMSE score for model: {statistics.mean(list_RMSE)}")


if __name__ == '__main__':
    Log.info(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    fittest_model(existing_model=True,
                  solution=[0.33453974, 0.42341855, 0.86770103, 0.466438, 0.63439439, 0.03518198, 0.69187014,
                            0.75762833],
                  model_path="FireflyAlgorithm_497dec739e724234ba2b68e2e29e659c4033b73b.pt")

    Log.info(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
