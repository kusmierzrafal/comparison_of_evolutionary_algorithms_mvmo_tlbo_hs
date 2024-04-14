import logging

from evolutionary_algorithms.evolutionary.crossover import Crossover
from evolutionary_algorithms.evolutionary.evolutionary_algorithm import EvolutionaryAlgorithm
from evolutionary_algorithms.evolutionary.mutation import Mutation
from evolutionary_algorithms.evolutionary.population import Population


class HS(EvolutionaryAlgorithm):
    def __init__(
        self,
        pcr: float,
        mutation_size: float = 0.25,
        mutation_factor: float = 0.2,
    ):
        """
        Harmony Search Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param pcr: ranges from 0.0 to 1.0
        :type pcr: float
        """
        logging.basicConfig(filename="hs.log", filemode="a", format="%(message)s")

        super().__init__()
        self.pcr = pcr
        self.mutation_factor = mutation_factor
        self.mutation_size = mutation_size
        self.crossover = Crossover(
            "one_from_population",
            population_considering_rate=self.pcr,
        )
        self.mutation = Mutation(
            "one_from_population",
            mutation_factor=self.mutation_factor,
            mutation_size=self.mutation_size,
        )

    def optimize(
        self,
        population: Population,
        iterations: int,
        optimize_function: callable,
        opt_val,
    ):
        self.crossover.init_population_based_parameters(population)
        self.mutation.init_population_based_parameters(population)
        super().init_population_based_parameters(population, iterations)

        population.evaluate(optimize_function)

        for iteration in range(iterations):
            child = self.crossover.cross(population)
            child = self.mutation.mutate(child)
            child_val = optimize_function(child)
            population.update_population(child, child_val)

            best_val = population.get_best_value()

            print(best_val)
            if super().termination_criterion(best_val, opt_val, iteration):
                return best_val
        return best_val
