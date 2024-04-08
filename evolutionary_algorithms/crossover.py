import copy
from functools import partial
from random import choice, random

import numpy as np

from evolutionary_algorithms.helpers import vstack
from evolutionary_algorithms.population import Population


class Crossover:

    def __init__(
        self,
        crossover_type,
    ):

        crossover_type_dict = {
            "mapping_transformation": self.init_mapping_crossover,
            "mean_difference_vector": self.init_difference_vector_crossover,
        }
        self.cross = crossover_type_dict[crossover_type]()

    def init_difference_vector_crossover(self):
        self.init_population_based_parameters = (
            self.difference_vector_population_based_parameters
        )
        return self.difference_vector_crossover

    def difference_vector_population_based_parameters(self, population: Population):
        self.dimensions = population.get_dimensions()

    def difference_vector_crossover(
        self, population: Population, optimize_function: callable
    ):
        """
        As input gets population to crossover
        """

        crossed_population = copy.deepcopy(population)
        random_factor = np.array([random() for _ in range(self.dimensions)])

        other_values_indexes = self.get_other_values_indexes(population)

        to_cross = []
        for ind, indexes in zip(range(population.get_size()), other_values_indexes):
            if len(indexes) == 0:
                return
            else:
                ind_to_cross = choice(indexes)
                to_cross.append(ind_to_cross)

        is_better = population.evaluations < population.evaluations[to_cross]

        crossed_population.population.T[is_better] = crossed_population.population.T[
            is_better
        ] + random_factor * (
            crossed_population.population.T[is_better]
            - population.population[:, to_cross].T[is_better]
        )
        crossed_population.population.T[~is_better] = crossed_population.population.T[
            ~is_better
        ] + random_factor * (
            population.population[:, to_cross].T[~is_better]
            - crossed_population.population.T[~is_better]
        )
        crossed_population.ensure_boundaries()
        crossed_population.evaluate(optimize_function)

        population.get_better(crossed_population)

        population.evaluate(optimize_function)
        population.sort()

    def get_other_values_indexes(self, population: Population):

        get_other_values_indexes = partial(
            self._get_other_values_indexes, population.evaluations
        )
        get_other_values_indexes = np.vectorize(
            get_other_values_indexes, otypes=[np.ndarray]
        )
        other_values_indexes = get_other_values_indexes(population.evaluations)

        return other_values_indexes

    @staticmethod
    def _get_other_values_indexes(evaluations, val):
        return np.where(evaluations != val)[0]

    def init_mapping_crossover(self):

        return self.mapping_crossover

    @staticmethod
    def mapping_crossover(population: Population, mask):
        """
        As input gets population to crossover and mask telling which individuals to cross
        """
        population_transposed = population.population.T
        pop_size = population.get_size()
        best_ind_pop_size = vstack(population.get_best_archive_individual, pop_size)
        population_transposed[mask] = best_ind_pop_size[mask]
