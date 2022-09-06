import math
import torch
import yaml
import argparse
from tabulate import tabulate
from experiments.rnn_vae_experiment import RNNVAExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from dataloaders.time_series import TimeSeriesDataset
from pytorch_lightning.plugins import DDPPlugin
from niapy.algorithms.modified import *

from models import vae_models
from storage.database import SQLiteConnector
import uuid
from pathlib import Path
from niapy_extension.wrapper import *

RUN_UUID = uuid.uuid4().hex
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

config['logging_params']['save_dir'] += RUN_UUID + '/'
Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")

conn = SQLiteConnector(config['logging_params']['db_storage'], f"solution_{RUN_UUID}")
seed_everything(config['exp_params']['manual_seed'], True)

datamodule = TimeSeriesDataset(**config["data_params"], pin_memory=True)
datamodule.setup()


class RNNVAEAEArchitecture(ExtendedProblem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution, alg_name):
        print("=================================================================================================")
        print(f"ITERATION: {self.iteration}")
        print(f"SOLUTION : {solution}")
        self.iteration += 1

        model = vae_models[config['model_params']['name']](solution, **config)
        existing_entry = conn.get_entries(hash_id=model.hash_id)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            print(f"Model for this solution already exists")
            return fitness

        else:
            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                RMSE = int(9e10)
            else:
                experiment = RNNVAExperiment(model, config['exp_params'], config['model_params']['n_features'])
                config['trainer_params']['min_epochs'] = model.num_epochs
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'] + 'all_models/',
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                runner = Trainer(logger=tb_logger,
                                 #progress_bar_refresh_rate=0,
                                 auto_select_gpus=True,
                                 callbacks=[
                                     LearningRateMonitor(),
                                     ModelCheckpoint(save_top_k=1,
                                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                                     monitor="val_loss",
                                                     save_last=True),
                                     early_stop_callback,
                                 ],
                                 strategy=DDPPlugin(find_unused_parameters=False),
                                 **config['trainer_params'])

                print(f"======= Training {config['model_params']['name']} =======")
                print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                runner.fit(experiment, datamodule=datamodule)
                print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

                # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
                if math.isnan(experiment.test_RMSE.item()):
                    RMSE = int(9e10)
                else:
                    RMSE = experiment.test_RMSE.item()

            complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
            fitness = (RMSE * 1000) + (complexity / 100)

            print(tabulate([[RMSE, complexity, fitness]], headers=["RMSE", "Complexity", "Fitness"], tablefmt="pretty"))
            conn.post_entries(model, fitness, solution, RMSE, complexity, alg_name, self.iteration)

            return fitness


if __name__ == '__main__':
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    print(f"RUN UUID: {RUN_UUID}")
    """
    Dimensionality:
    y1: topology shape,
    y2: layer type
    y3: number of neurons per layer,
    y4: number of layers,
    y5: activation function
    y6: number of epochs,
    y7: learning rate
    y8: optimizer algorithm.
    """
    DIMENSIONALITY = 8

    runner = ExtendedRunner(
        config['logging_params']['save_dir'],
        dimension=DIMENSIONALITY,
        max_evals=100,
        runs=2,
        algorithms=[
            ParticleSwarmAlgorithm(),
            DifferentialEvolution(),
            FireflyAlgorithm(),
            SelfAdaptiveDifferentialEvolution(),
            GeneticAlgorithm()
        ],
        problems=[
            RNNVAEAEArchitecture(DIMENSIONALITY)
        ]
    )

    print("=====================================SEARCH STARTED==============================================")
    final_solutions = runner.run(export='json', verbose=True)
    print("=====================================SEARCH COMPLETED============================================")

    best_solution, best_algorithm = conn.best_results()
    best_model = vae_models[config['model_params']['name']](best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{best_algorithm}_{best_model.hash_id}.pt"
    torch.save(best_model, model_file)
    print(f"Best model saved to: {model_file}")
    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
