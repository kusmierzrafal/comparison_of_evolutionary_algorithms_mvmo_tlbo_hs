import logging

from evolutionary_algorithms.evolutionary.crossover import Crossover
from evolutionary_algorithms.evolutionary.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
)
from evolutionary_algorithms.evolutionary.mutation import Mutation
from evolutionary_algorithms.evolutionary.population import Population


class MVMO(EvolutionaryAlgorithm):
    """
    Implementation of Mean-Variance Optimization Algorithm
    """

    def __init__(
        self,
        mutation_size: int,
        n_best_size: int = 10,
        shaping_scaling_factor_fs=1.0,
        asymmetry_factor_af=1.0,
        val_shape_factor_sd=75.0,
    ):
        """
        :param mutation_size: number of genes to be mutated
        :param n_best_size: number of individuals to be archived in best population
        :param shaping_scaling_factor_fs: float number between 0.9 and 1.0
            for exploration, between 1.0 and 10.0 for exploitation
        :param asymmetry_factor_af: float number between 1.0 and 10.0
        :param val_shape_factor_sd: float number between 10.0 and 90.0
        """

        logging.basicConfig(filename="mvmo.log", filemode="a", format="%(message)s")

        super().__init__()
        self.shaping_scaling_factor_fs = shaping_scaling_factor_fs
        self.asymmetry_factor_af = asymmetry_factor_af
        self.val_shape_factor_sd = val_shape_factor_sd
        self.n_best_size = n_best_size
        self.mutation = Mutation(
            "mapping_transformation",
            mutation_size=mutation_size,
            shaping_scaling_factor_fs=shaping_scaling_factor_fs,
            asymmetry_factor_af=asymmetry_factor_af,
            val_shape_factor_sd=val_shape_factor_sd,
        )
        self.crossover = Crossover(
            "mapping_transformation",
        )

    def optimize(
        self,
        population: Population,
        iterations: int,
        optimize_function: callable,
        opt_val,
    ):
        population.archive_best_population(self.n_best_size, optimize_function)
        population.normalize()
        self.mutation.init_population_based_parameters(population)
        self.crossover.init_population_based_parameters(population)
        super().init_population_based_parameters(population, iterations)

        for iteration in range(iterations):
            child, mutation_indexes = self.mutation.mutate(population)
            child = self.crossover.cross(child, mutation_indexes, population)
            child_val = population.evaluate_denormalized_ind(child, optimize_function)
            population.update_best_population(self.n_best_size, child, child_val)
            best_val = population.get_best_value()

            if super().termination_criterion(best_val, opt_val, iteration):
                return best_val
        return best_val
