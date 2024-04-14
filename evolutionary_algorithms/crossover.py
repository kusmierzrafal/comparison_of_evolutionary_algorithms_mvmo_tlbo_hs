import copy
from functools import partial
from random import choice, randint, random, uniform

import numpy as np

from evolutionary_algorithms.helpers import vstack
from evolutionary_algorithms.population import Population


class Crossover:

    def __init__(self, crossover_type, **kwargs):

        crossover_type_dict = {
            "mapping_transformation": self.init_mapping_crossover,
            "mean_difference_vector": self.init_difference_vector_crossover,
            "one_from_population": self.init_one_from_population_crossover,
        }
        self.cross = crossover_type_dict[crossover_type](kwargs)

    def init_difference_vector_crossover(self, kwargs):
        self.init_population_based_parameters = (
            self.difference_vector_population_based_parameters
        )
        return self.difference_vector_crossover

    def difference_vector_population_based_parameters(self, population: Population):
        self.dimensions = population.get_dimensions()

    def one_from_population_population_based_parameters(self, population: Population):
        self.dimensions = population.get_dimensions()
        self.boundaries = population.get_boundaries()
        self.size = population.get_size()

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

    def init_mapping_crossover(self, kwargs):

        return self.mapping_crossover

    def init_one_from_population_crossover(self, kwargs):
        self.init_population_based_parameters = (
            self.one_from_population_population_based_parameters
        )

        self.pcr = kwargs["population_considering_rate"]
        return self.one_from_population_crossover

    @staticmethod
    def mapping_crossover(population: Population, mask):
        """
        As input gets population to crossover and mask telling which individuals to cross
        """
        population_transposed = population.population.T
        pop_size = population.get_size()
        best_ind_pop_size = vstack(population.get_best_archive_individual, pop_size)
        population_transposed[mask] = best_ind_pop_size[mask]

    def one_from_population_crossover(self, population: Population):
        transposed_population = population.population.T
        child = np.empty((self.dimensions, 1), dtype=float)
        for ind in range(self.dimensions):
            if random() < self.pcr:
                ind_index = randint(0, self.size - 1)
                child[ind] = transposed_population[ind_index][ind]
            else:
                child[ind] = uniform(self.boundaries[0], self.boundaries[1])
        return child
