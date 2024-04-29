import logging
from abc import ABC, abstractmethod

from evolutionary_algorithms.evolutionary.crossover import Crossover
from evolutionary_algorithms.evolutionary.mutation import Mutation
from evolutionary_algorithms.evolutionary.population import Population


class EvolutionaryAlgorithm(ABC):

    def __init__(
        self,
        algorithm,
        **kwargs,
    ):
        self.error_value = 10**-8
        self.logging_times = 16
        self.k = 0

        self.mutation = Mutation(
            algorithm,
            **kwargs,
        )
        self.crossover = Crossover(
            algorithm,
            **kwargs,
        )

    @abstractmethod
    def optimize(
        self,
        population: Population,
        iterations: int,
        optimize_function: callable,
        opt_val: int,
    ):
        """
        Newly created algorithm inheriting from this class has to implement optimize function.
        In its optimize implementation should be prepared whole flow of implemented algorithm.
        To achive this, usage of population, mutation and crossover classes might be helpful.
        These classes provide useful mechanisms usually used in evolutionary algorithms.
        Their goal is to enable building new algorithms from already prepared components.
        In addition, their implementations use numpy so they work in optimal way.
        """
        pass

    def termination_criterion(self, best_val, opt_val, iteration):
        diff = best_val - opt_val
        if diff < self.error_value:
            logging.warning(f"{iteration + 1}")
            return True
        if iteration + 1 >= self.k_FES[self.k]:
            logging.warning(f"{diff}")
            self.k = self.k + 1

        return False

    def init_population_based_parameters(
        self, population: Population, max_iterations: int
    ):
        dimensions = population.get_dimensions()
        self.k_FES = {
            k: dimensions ** (k / 5 - 3) * max_iterations
            for k in range(self.logging_times)
        }

        self.mutation.init_population_based_parameters(population)
        self.crossover.init_population_based_parameters(population)
