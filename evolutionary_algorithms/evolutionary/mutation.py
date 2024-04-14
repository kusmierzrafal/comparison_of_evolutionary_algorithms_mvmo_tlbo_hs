import copy
import math
import random

import numpy as np

from evolutionary_algorithms.evolutionary.helpers import vstack
from evolutionary_algorithms.evolutionary.population import Population


class Mutation:

    def __init__(
        self,
        mutation_type,
        **kwargs,
    ):

        mutation_type_dict = {
            "mapping_transformation": self.init_mapping_mutation,
            "mean_difference_vector": self.init_difference_vector_mutation,
            "one_from_population": self.init_one_from_population_mutation,
        }
        self.mutate = mutation_type_dict[mutation_type](kwargs)

    def mapping_population_based_parameters(self, population: Population):
        self.dimensions = population.get_dimensions()
        self.pop_size = population.get_size()
        self.kd = 0.0505 / self.dimensions + 1.0
        self.last_no_zero_var_ind = np.ones(self.dimensions)

    def one_from_population_population_based_parameters(self, population: Population):
        self.dimensions = population.get_dimensions()
        self.boundaries = population.get_boundaries()

    def difference_vector_population_based_parameters(self, population: Population):
        self.boundaries = population.get_boundaries()
        self.dimensions = population.get_dimensions()
        self.pop_size = population.get_size()

    def init_mapping_mutation(self, kwargs):

        self.mutation_size = kwargs["mutation_size"]
        self.shaping_scaling_factor_fs = kwargs["shaping_scaling_factor_fs"]
        self.asymmetry_factor_af = kwargs["asymmetry_factor_af"]
        self.val_shape_factor_sd = kwargs["val_shape_factor_sd"]
        self.current_mutation_position = 0
        self.init_population_based_parameters = self.mapping_population_based_parameters
        return self.mapping_mutation

    def init_one_from_population_mutation(self, kwargs):
        self.mutation_factor = kwargs["mutation_factor"]
        self.mutation_size = kwargs["mutation_size"]
        self.init_population_based_parameters = (
            self.one_from_population_population_based_parameters
        )
        return self.one_from_population_mutation

    def init_difference_vector_mutation(self, kwargs):

        self.teaching_factor = random.randint(1, 2)
        self.init_population_based_parameters = (
            self.difference_vector_population_based_parameters
        )
        return self.difference_vector_mutation

    def difference_vector_mutation(
        self, population: Population, optimize_function: callable
    ):

        mutated_population = copy.deepcopy(population)
        mean_individual = mutated_population.get_mean_individual()
        best_individual = mutated_population.get_best_individual()

        random_factor = np.array([random.random() for _ in range(self.dimensions)])
        mutagen = random_factor * (
            best_individual - self.teaching_factor * mean_individual
        )
        mutagen_pop_size = np.vstack([mutagen] * self.pop_size)

        mutated_population_transposed = mutated_population.population.T
        mutated_population_transposed += mutagen_pop_size

        mutated_population.ensure_boundaries()
        mutated_population.evaluate(optimize_function)

        population.get_better(mutated_population)

    def _last_no_zero_var_ind(self, new_var):
        self.last_no_zero_var_ind = np.asarray(
            [
                new if (new != 0 and np.isfinite(new)) else last_non_zero
                for (new, last_non_zero) in zip(new_var, self.last_no_zero_var_ind)
            ]
        )
        return self.last_no_zero_var_ind

    def get_mutation_indexes(self):
        mut_ind = (
            np.asarray(
                range(
                    self.current_mutation_position,
                    self.current_mutation_position + self.mutation_size,
                )
            )
            % self.dimensions
        )
        self.current_mutation_position += self.mutation_size
        return mut_ind

    def mask_ind(self, ind: np.ndarray):
        ind[self.get_mutation_indexes()] = True

    def mutation_mask(self, size):
        mask = np.full((size, self.dimensions), False)
        np.apply_along_axis(self.mask_ind, 1, mask)
        return mask

    def mapping_matrix(
        self, population: Population, population_transposed, mask, pop_size
    ):
        var_individual = population.get_var_archive_individual()
        self._last_no_zero_var_ind(var_individual)
        var_ind_pop_size = np.vstack([var_individual] * pop_size)
        last_no_zero_var_ind_pop_size = np.vstack(
            [self.last_no_zero_var_ind] * pop_size
        )

        mean_ind_pop_size = vstack(population.get_mean_archive_individual, pop_size)
        best_ind_pop_size = vstack(population.get_best_archive_individual, pop_size)

        return best_ind_pop_size, zip(
            population_transposed[mask],
            mean_ind_pop_size[mask],
            best_ind_pop_size[mask],
            var_ind_pop_size[mask],
            last_no_zero_var_ind_pop_size[mask],
        )

    def mapping_mutate(self, mapping_matrix):
        mutated = []

        for ind, mean_ind, best_ind, var_ind, last_no_zero_var in mapping_matrix:
            si, si1, si2 = self.count_si(
                best_ind,
                mean_ind,
                var_ind,
                last_no_zero_var,
            )
            ind = self.transformation(
                np.random.uniform(low=0, high=1, size=(1,))[0],
                mean_ind,
                si1,
                si2,
            )
            mutated.append(ind)
        mutated = np.asarray(mutated)
        return mutated

    def mapping_mutation(
        self,
        population: Population,
    ):
        """
        Strategy 2a used
        """

        population_transposed = population.population.T

        mask = self.mutation_mask(self.pop_size)
        best_ind_pop_size, mapping_matrix = self.mapping_matrix(
            population, population_transposed, mask, self.pop_size
        )

        mutated = self.mapping_mutate(mapping_matrix)
        population_transposed[mask] = mutated

        return ~mask

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

    def one_from_population_mutation(self, child: np.ndarray):

        for ind in range(self.dimensions):
            if random.random() < self.mutation_factor:
                if random.random() < 0.5:
                    child[ind] -= (
                        (child[ind] - self.boundaries[0])
                        * random.random()
                        * self.mutation_size
                    )
                else:
                    child[ind] += (
                        (self.boundaries[1] - child[ind])
                        * random.random()
                        * self.mutation_size
                    )
        return child
