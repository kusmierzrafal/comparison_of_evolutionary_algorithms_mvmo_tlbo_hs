import copy
import logging
import math
import random

import numpy as np

from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization_functions.optimization_functions import zakharov_function


class MVMO(EvolutionaryAlgorithm):
    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
        mutation_size: int,
        shaping_scaling_factor_fs=1.0,
        asymmetry_factor_af=1.0,
        val_shape_factor_sd=75.0,
    ):
        """
        A Mean-Variance Optimization Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param mutation_size: number of genes to be mutated
        :type mutation_size: int
        :param shaping_scaling_factor_fs: between 0.9 and 1.0
            for exploration, between 1.0 and 10.0 for exploitation
        :type shaping_scaling_factor_fs: float
        :param asymmetry_factor_af: between 1.0 and 10.0
        :type asymmetry_factor_af: float
        :param val_shape_factor_sd: between 10.0 and 90.0, by default 75.0
        :type val_shape_factor_sd: float
        """
        logging.basicConfig(filename="mvmo.log", filemode="a", format="%(message)s")

        super().__init__(iterations, dimensions, boundaries)
        self.mutation_size = mutation_size
        self.shaping_scaling_factor_fs = shaping_scaling_factor_fs
        self.asymmetry_factor_af = asymmetry_factor_af
        self.val_shape_factor_sd = val_shape_factor_sd
        self.kd = 0.0505 / self.dimensions + 1.0
        self.n_best_size = 10
        self.current_mutation_position = 0
        self.last_no_zero_var_ind = np.ones(self.dimensions)

    def optimize(self, population: list[np.ndarray], optimize_function: callable):
        """

        :param population:
        :type population:
        :param optimize_function:
        :type optimize_function:
        :return:
        :rtype:
        """
        normalized_population = self.normalize_population(population)
        best_population = best_individual = None

        for i in range(self.iterations):
            best_population, mean_individual, var_individual = self.evaluation(
                normalized_population,
                optimize_function,
                self.n_best_size,
                best_population,
            )
            if best_individual is None:
                best_individual = best_population[0]
            elif best_population[0][1] < best_individual[1]:
                best_individual = best_population[0]

            normalized_population = self.mutation(
                normalized_population,
                mean_individual,
                var_individual,
                best_individual[0],
            )

        return best_individual

    def normalize_population(self, population: list[np.ndarray]):
        """
        Normalizes the input population
        :param population: population to normalize
        :return: normalized population (list)
        """
        return [
            (ind - self.boundaries[0]) / (self.boundaries[1] - self.boundaries[0])
            for ind in population
        ]

    def denormalize_population(self, population: list[np.ndarray]):
        """
        Denormalizes the input population
        :param population: population to denormalize
        :return: denormalized population (list)
        """
        return [
            (ind * (self.boundaries[1] - self.boundaries[0])) + self.boundaries[0]
            for ind in population
        ]

    @staticmethod
    def transformation(random_gene, mean_gene, si1, si2):
        def transform(ui):
            return mean_gene * (1 - math.exp(-1 * ui * si1)) + (
                1 - mean_gene
            ) * math.exp((ui - 1) * si2)

        return (
            transform(random_gene)
            + (1 - transform(1) + transform(0)) * random_gene
            - transform(0)
        )

    def count_si(self, best_gene, mean_gene, var_gene, last_no_zero_var_gene):

        if not np.isfinite(var_gene) or var_gene == 0:
            last_no_zero_si = (
                -1 * np.log(last_no_zero_var_gene) * self.shaping_scaling_factor_fs
            )
            si = si1 = si2 = last_no_zero_si
            if si < self.val_shape_factor_sd:
                self.val_shape_factor_sd = self.val_shape_factor_sd * self.kd
                si1 = self.val_shape_factor_sd
            elif si > self.val_shape_factor_sd:
                self.val_shape_factor_sd = self.val_shape_factor_sd / self.kd
                si1 = self.val_shape_factor_sd
        else:
            si = -1 * np.log(var_gene) * self.shaping_scaling_factor_fs
            si1 = si
            si2 = si

        if best_gene < mean_gene:
            si2 = si * self.asymmetry_factor_af
        elif best_gene > mean_gene:
            si1 = si * self.asymmetry_factor_af

        return si, si1, si2

    def mutation(
        self,
        population: list[np.ndarray],
        mean_individual: np.ndarray,
        var_individual: np.ndarray,
        best_individual: np.ndarray,
    ):
        """
        Strategy 2a used
        :param population:
        :type population:
        :param mean_individual:
        :type mean_individual:
        :param var_individual:
        :type var_individual:
        :param best_individual:
        :type best_individual:
        :return:
        :rtype:
        """
        population = copy.deepcopy(population)
        for individual in population:
            for _ in range(self.mutation_size):
                ind = self.current_mutation_position % self.dimensions
                si, si1, si2 = self.count_si(
                    best_individual[ind],
                    mean_individual[ind],
                    var_individual[ind],
                    self.last_no_zero_var_ind[ind],
                )
                individual[ind] = self.transformation(
                    np.random.uniform(low=0, high=1, size=(1,))[0],
                    mean_individual[ind],
                    si1,
                    si2,
                )
                self.current_mutation_position += 1
            for i in range(self.dimensions - self.mutation_size):
                ind = (self.current_mutation_position + i) % self.dimensions
                individual[ind] = best_individual[ind]

        return population

    def evaluation(
        self,
        population: list[np.ndarray],
        fitness_function: callable,
        n_best_size: int = 2,
        best_population=None,
    ) -> tuple:

        # best_population = normalized individuals with
        # fitness values calculated for denormalized individuals
        denormalized_population = self.denormalize_population(population)

        evaluated_population = [
            (n_ind, fitness_function(dn_ind))
            for n_ind, dn_ind in zip(population, denormalized_population)
        ]

        if best_population is not None:
            evaluated_population = evaluated_population + best_population

        best_population = sorted(
            evaluated_population,
            key=lambda ind: ind[1],
        ).copy()[:n_best_size]

        mean_individual = np.mean([ind[0] for ind in best_population], axis=0)
        var_individual = np.var([ind[0] for ind in best_population], axis=0)
        self.last_no_zero_var_ind = np.asarray(
            [
                new if (new != 0 and np.isfinite(new)) else last_non_zero
                for (new, last_non_zero) in zip(
                    var_individual, self.last_no_zero_var_ind
                )
            ]
        )

        return best_population, mean_individual, var_individual


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    boundaries = (-5.12, 5.12)
    optimizer = MVMO(10, 6, boundaries, 3)
    population = optimizer.init_population(1000)
    optimizer.optimize(population, zakharov_function)
