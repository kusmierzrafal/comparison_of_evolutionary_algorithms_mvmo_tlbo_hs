import logging

import numpy as np

from evolutionary_algorithms.crossover import Crossover
from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from evolutionary_algorithms.mutation import Mutation


class TLBO(EvolutionaryAlgorithm):
    def __init__(
        self,
    ):
        """
        Teaching Learning Based Optimization Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        """
        self.mutation = Mutation(
            "mean_difference_vector",
        )
        self.crossover = Crossover(
            "mean_difference_vector",
        )
        logging.basicConfig(filename="tlbo.log", filemode="a", format="%(message)s")

        super().__init__()

    def optimize(
        self,
        population: list[np.ndarray],
        iterations: int,
        optimize_function: callable,
        opt_val,
    ):
        """
        Searches for the best solution for a given number of iterations
        :param population: initial population
        :type population: list[np.ndarray]
        :param optimize_function:
        :type optimize_function: callable
        :return: best from found solutions
        :rtype: numpy.ndarray
        """

        self.mutation.init_population_based_parameters(population)
        self.crossover.init_population_based_parameters(population)
        super().init_population_based_parameters(population, iterations)

        population.evaluate(optimize_function)

        for iteration in range(iterations):

            self.mutation.mutate(population, optimize_function)
            self.crossover.cross(population, optimize_function)

            best_val = population.get_best_value()

            if super().termination_criterion(best_val, opt_val, iteration):
                return best_val
        return best_val
