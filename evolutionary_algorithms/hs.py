from evolutionary_algorithms.evolutionary.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
)
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

        super().__init__(
            "one_from_population",
            population_considering_rate=pcr,
            mutation_factor=mutation_factor,
            mutation_size=mutation_size,
        )

    def optimize(
        self,
        population: Population,
        iterations: int,
        optimize_function: callable,
        opt_val,
        result_file,
    ):
        super().init_population_based_parameters(population, iterations)

        population.evaluate(optimize_function)

        for iteration in range(iterations):
            child = self.crossover.cross(population)
            child = self.mutation.mutate(child)
            child_val = optimize_function(child)
            population.update_population(child, child_val)

            best_val = population.get_best_value()

            if super().termination_criterion(best_val, opt_val, iteration, result_file):
                return best_val
        return best_val
