import logging
import random

import numpy as np

from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization_functions.optimization_functions import rastrigins_function


class HS(EvolutionaryAlgorithm):
    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
        maximize: bool,
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
        :param maximize: True for maximization, False for minimization
        :type maximize: bool
        :param hmcr: ranges from 0.0 to 1.0
        :type hmcr: float
        """
        logging.basicConfig(filename="hs.log", filemode="a", format="%(message)s")

        super().__init__(iterations, dimensions, boundaries, maximize)
        self.hmcr = hmcr

    def evaluation(
        self,
        population: list[np.ndarray],
        fitness_function: callable,
        child: np.ndarray,
    ):
        population = population + [child]

        best_population = sorted(
            [(ind, fitness_function(ind)) for ind in population],
            key=lambda ind: ind[1],
            reverse=self.maximize,
        ).copy()[: len(population) - 1]
        return best_population

    def reproduction(self, population: list[np.ndarray]) -> np.ndarray:
        child = np.empty(self.dimensions, dtype=float)
        for ind in range(self.dimensions):
            if random.random() > self.hmcr:
                child[ind] = random.uniform(self.boundaries[0], self.boundaries[1])
            else:
                child[ind] = random.choice(population)[ind]

        return child

    def optimize(self, population: list[np.ndarray], optimize_function: callable):

        for i in range(self.iterations):

            child = self.reproduction(population)
            evaluated_population = self.evaluation(population, optimize_function, child)

            population = [ind[0] for ind in evaluated_population]

        return evaluated_population[0]


if __name__ == "__main__":
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, 6, boundaries, True, hmcr=0.9)
    population = optimizer.init_population(100)
    optimizer.optimize(population, rastrigins_function)
