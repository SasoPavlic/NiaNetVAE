import os
import statistics
import yaml
import argparse
from sklearn.metrics import mean_squared_error
from models import *
from experiments.rnn_vae_experiment import RNNVAExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from datasets.time_series import TimeSeriesDataset
from pytorch_lightning.plugins import DDPPlugin

parser = argparse.ArgumentParser(description='Generic runner for LSTM VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/rnn_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


def fittest_model(existing_model, **kwargs):
    dataloader = TimeSeriesDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    dataloader.setup()

    if existing_model:
        model = torch.load(kwargs["model_path"])
    else:
        model = vae_models[config['model_params']['name']](kwargs["solution"], **config['model_params'])
        experiment = RNNVAExperiment(model, config['exp_params'], config['model_params']['n_features'])
        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True),
                             early_stop_callback,
                         ],
                         strategy=DDPPlugin(find_unused_parameters=False),

                         **config['trainer_params'])

        print(f"======= Training {config['model_params']['name']} =======")

        print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
        runner.fit(experiment, datamodule=dataloader)
        print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

    dataloader_iterator = iter(dataloader.test_dataloader())
    list_RMSE = list()
    for i in range(448):
        try:
            data, target = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            data, target = next(dataloader_iterator)

        predictions = model(data)[0].detach().numpy()
        RMSE = mean_squared_error(data, predictions, squared=False)
        list_RMSE.append(RMSE)

    print(f"Mean RMSE score for model: {statistics.mean(list_RMSE)}")


if __name__ == '__main__':
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    fittest_model(existing_model=True,
                  solution=[0.19027519, 0.4210529, 0.92704726, 0.34496538, 0.74664277, 0.11013242, 0.40970904,
                            0.04878359],
                  model_path=f"{'ParticleSwarmAlgorithm'}_{'1659712733'}.pt")

    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
