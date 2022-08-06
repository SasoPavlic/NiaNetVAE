import math
import sys
import pandas as pd
import yaml
import argparse
from tabulate import tabulate
from peewee import SqliteDatabase

import evaluate
import storage.database
from models import *
from experiments.rnn_vae_experiment import RNNVAExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from datasets.time_series import TimeSeriesDataset
from pytorch_lightning.plugins import DDPPlugin
from niapy import Runner
from niapy.problems import Problem
from niapy.algorithms.basic import *
from niapy.algorithms.modified import *
import sqlite3 as sq

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

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

db = SqliteDatabase(config['logging_params']['db_storage'])
table_name = "solution"

class RNNVAEAEArchitecture(Problem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution):
        # solution = [0.38641122, 0.02673898, 0.55739414, 0.96943802, 0.67513284, 0.10191641, 0.6720203, 0.94043456]

        print("=================================================================================================")
        print(f"ITERATION IS: {self.iteration}")
        print(f"SOLUTION: {solution}")
        self.iteration += 1

        model = vae_models[config['model_params']['name']](solution, **config['model_params'])
        conn = sq.connect(f'{table_name}.sqlite')
        existing_entry = pd.read_sql(f"select * from {table_name} where hash_id='{model.hash_id}'", conn)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            conn.close()
            return fitness

        else:

            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                fitness = sys.maxsize
                print(f"Fitness: {fitness}")
                return fitness

            experiment = RNNVAExperiment(model, config['exp_params'], config['model_params']['n_features'])
            data = TimeSeriesDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
            config['trainer_params']['max_epochs'] = model.num_epochs
            data.setup()

            tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                          name=config['model_params']['name'] + model.hash_id,
                                          )

            runner = Trainer(logger=tb_logger,
                             progress_bar_refresh_rate=0,
                             weights_summary=None,
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
            runner.fit(experiment, datamodule=data)
            print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

            # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
            if math.isnan(experiment.val_RMSE.item()):
                RMSE = sys.maxsize
                complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
                fitness = sys.maxsize
                print(f"Fitness: {fitness}")

            else:
                RMSE = experiment.val_RMSE.item()
                complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
                fitness = (RMSE * 1000) + (complexity / 100)
                print(f"RMSE: {RMSE}")
                print(f"Complexity: {complexity}")
                print(f"Fitness: {fitness}")

            # TODO add solution array and algorithm name
            # TODO move to database.py
            df = pd.DataFrame({'hash_id': model.hash_id,
                               'timestamp': datetime.now().strftime("%H:%M %d-%m-%Y"),
                               'encoding_layers': str(model.encoding_layers),
                               'decoding_layers': str(model.decoding_layers),
                               'topology_shape': model.topology_shape,
                               'layer_type': model.layer_type,
                               'num_layers': model.num_layers,
                               'activation': str(model.activation_name),
                               'num_epochs': model.num_epochs,
                               'learning_rate': model.learning_rate,
                               'optimizer': str(model.optimizer_name),
                               'bottleneck_size': model.bottleneck_size,
                               'RMSE': RMSE,
                               'complexity': complexity,
                               'fitness': fitness}, index=[0])
            df.to_sql(table_name, conn, if_exists='append', index=False)  # writes to file
            conn.close()  # good practice: close connection

            return fitness


if __name__ == '__main__':
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
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

    runner = Runner(
        dimension=DIMENSIONALITY,
        max_evals=20,
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

    print("=================================================================================================")
    final_solutions = runner.run(export='json', verbose=True)
    print("=================================================================================================")
    best_fitness = sys.maxsize
    best_solution = None
    best_algorithm = None
    outputs = []

    for algorithm in final_solutions:
        fitness = final_solutions[algorithm]['RNNVAEAEArchitecture'][0][1]
        solution = str(final_solutions[algorithm]['RNNVAEAEArchitecture'][0][0]).strip()

        outputs.append([algorithm, fitness, solution])

        if best_fitness > fitness:
            best_fitness = fitness
            best_algorithm = algorithm
            best_solution = final_solutions[algorithm]['RNNVAEAEArchitecture'][0][0]

    print(tabulate(outputs, headers=["Algorithm", "Fitness", "Solution"]))
    print("=================================================================================================")
    print(f"Best algorithm: {best_algorithm}")
    print(f"Best solution: {best_solution}")
    best_model = vae_models[config['model_params']['name']](best_solution, **config['model_params'])
    model_file = f"{best_algorithm}_{best_model.hash_id}.pt"
    torch.save(best_model, model_file)

    evaluate.fittest_model(existing_model=True,
                           solution=best_solution,
                           model_path=model_file)

    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
