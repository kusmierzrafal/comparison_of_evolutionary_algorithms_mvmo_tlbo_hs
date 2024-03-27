import logging
import random
import bisect

import numpy as np

from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization_functions.optimization_functions import zakharov_function


class HS(EvolutionaryAlgorithm):
    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
        hmcr: float,
    ):
        """
        Harmony Search Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param hmcr: ranges from 0.0 to 1.0
        :type hmcr: float
        """
        logging.basicConfig(filename="hs.log", filemode="a", format="%(message)s")

        super().__init__(iterations, dimensions, boundaries)
        self.hmcr = hmcr

    def evaluation(
        self,
        population: list[np.ndarray],
        fitness_function: callable,
    ):

        evaluated_population = sorted(
            [(ind, fitness_function(ind)) for ind in population],
            key=lambda ind: ind[1],
        )

        return evaluated_population

    def reproduction(self, population: list[np.ndarray]) -> np.ndarray:
        child = np.empty(self.dimensions, dtype=float)
        for ind in range(self.dimensions):
            if random.random() > self.hmcr:
                child[ind] = random.uniform(self.boundaries[0], self.boundaries[1])
            else:
                child[ind] = random.choice(population)[ind]

        return child

    def optimize(self, population: list[np.ndarray], optimize_function: callable):

        evaluated_population = self.evaluation(population, optimize_function)
        for i in range(self.iterations):
            child = self.reproduction(population)
            evaluated_child = (child, optimize_function(child))
            bisect.insort(evaluated_population, evaluated_child, key=lambda ind: ind[1])
            evaluated_population = evaluated_population[:-1]
        return evaluated_population[0]


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)
    boundaries = (-10, 10)
    optimizer = HS(100000, 6, boundaries, hmcr=0.4)
    population = optimizer.init_population(10)
    optimizer.optimize(population, zakharov_function)
