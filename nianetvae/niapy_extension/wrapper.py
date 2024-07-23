import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from niapy import Runner
from niapy.algorithms import Algorithm
from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from niapy.util import limit
from niapy.util.factory import get_algorithm

logging.basicConfig()
logger = logging.getLogger("niapy.task.Task")
logger.setLevel("INFO")


class ExtendedProblem(Problem):

    def __init__(self, dimension=1, lower=None, upper=None, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    def _evaluate(self, x, alg_name):
        pass

    def evaluate(self, x, alg_name):
        if x.shape[0] != self.dimension:
            raise ValueError('Dimensions do not match. {} != {}'.format(x.shape[0], self.dimension))

        return self._evaluate(x, alg_name)


class ExtendedTask(Task):

    def __init__(self, alg_name, problem=None, dimension=None, lower=None, upper=None,
                 optimization_type=OptimizationType.MINIMIZATION, repair_function=limit, max_evals=np.inf,
                 max_iters=np.inf, cutoff_value=None, enable_logging=False):

        self.alg_name = alg_name
        super().__init__(problem, dimension, lower, upper, optimization_type, repair_function, max_evals, max_iters,
                         cutoff_value, enable_logging)

    def eval(self, x):
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.

        """
        if self.stopping_condition():
            return np.inf

        self.evals += 1
        x_f = self.problem.evaluate(x, self.alg_name) * self.optimization_type.value

        if x_f < self.x_f * self.optimization_type.value:
            self.x_f = x_f * self.optimization_type.value
            self.n_evals.append(self.evals)
            self.fitness_evals.append(x_f)
            if self.enable_logging:
                logger.info('evals:%d => %s' % (self.evals, self.x_f))
        return x_f


class ExtendedRunner(Runner):

    def __init__(self, dir_path, dimension=10, max_evals=1000000, runs=1, algorithms='ArtificialBeeColonyAlgorithm',
                 problems='Ackley', optimization_type=OptimizationType.MINIMIZATION):

        self.optimization_type = optimization_type
        self.dir_path = dir_path
        super().__init__(dimension, max_evals, runs, algorithms, problems)

    def task_factory(self, alg_name, name):
        return ExtendedTask(alg_name, max_evals=self.max_evals, dimension=self.dimension, problem=name, optimization_type=self.optimization_type)

    def __create_export_dir(self):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def __generate_export_name(self, extension):
        self.__create_export_dir()
        return self.dir_path + str(datetime.now()).replace(":", ".") + "." + extension

    def __export_to_json(self):
        dataframe = pd.DataFrame.from_dict(self.results)
        dataframe.to_json(self.__generate_export_name("json"))
        logger.info("Export to JSON file completed!")

    def run(self, export="dataframe", verbose=False):
        """Execute runner.

        Args:
            export (str): Takes export type (e.g. dataframe, json, xls, xlsx) (default: "dataframe")
            verbose (bool): Switch for verbose logging (default: {False})

        Returns:
            dict: Returns dictionary of results

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        for alg in self.algorithms:
            if not isinstance(alg, "".__class__):
                alg_name = str(type(alg).__name__)
            else:
                alg_name = alg

            self.results[alg_name] = {}

            if verbose:
                logger.info("Running %s...", alg_name)

            for problem in self.problems:
                if not isinstance(problem, "".__class__):
                    problem_name = str(type(problem).__name__)
                else:
                    problem_name = problem

                if verbose:
                    logger.info("Running %s algorithm on %s problem...", alg_name, problem_name)

                self.results[alg_name][problem_name] = []
                for _ in range(self.runs):
                    if isinstance(alg, Algorithm):
                        algorithm = alg
                    else:
                        algorithm = get_algorithm(alg)
                    task = self.task_factory(alg_name, problem)
                    self.results[alg_name][problem_name].append(algorithm.run(task))
            if verbose:
                logger.info("---------------------------------------------------")
        if export == "dataframe":
            self.__export_to_dataframe_pickle()
        elif export == "json":
            self.__export_to_json()
        elif export == "xsl":
            self._export_to_xls()
        elif export == "xlsx":
            self.__export_to_xlsx()
        else:
            raise TypeError("Passed export type %s is not supported!", export)
        return self.results
