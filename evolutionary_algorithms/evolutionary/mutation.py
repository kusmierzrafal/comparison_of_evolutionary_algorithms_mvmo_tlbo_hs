import copy
import math
import random

import numpy as np

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
        self.last_no_zero_var_ind = np.where(
            (np.isfinite(new_var)) & (new_var != 0), new_var, self.last_no_zero_var_ind
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

    def mapping_mutation(
        self,
        population: Population,
    ):
        """
        Strategy 2a used
        """
        child = np.empty(self.dimensions, dtype=float)
        mutation_indexes = self.get_mutation_indexes()
        mean_individual = population.get_mean_individual()
        # rounded in order to avoid minimal differences during comparison connected with rounding
        mean_individual = np.round(mean_individual, 8)
        best_individual = population.get_best_individual()
        var_individual = population.get_var_individual()
        # rounded in order to avoid minimal differences during comparison connected with rounding
        var_individual = np.round(var_individual, 8)
        self._last_no_zero_var_ind(var_individual)
        last_no_zero_var_ind = self.last_no_zero_var_ind
        for ind in mutation_indexes:
            si, si1, si2 = self.count_si(
                best_individual[ind],
                mean_individual[ind],
                var_individual[ind],
                last_no_zero_var_ind[ind],
            )

            new_gene = self.transformation(
                np.random.uniform(low=0, high=1, size=(1,))[0],
                mean_individual[ind],
                si1,
                si2,
            )
            child[ind] = new_gene

        return child, mutation_indexes

    def count_si(self, best_gene, mean_gene, var_gene, last_no_zero_var_gene):

        if not np.isfinite(var_gene) or var_gene == 0:
            last_no_zero_si = (
                -1 * np.log(last_no_zero_var_gene) * self.shaping_scaling_factor_fs
            )
            si = si1 = si2 = last_no_zero_si
            # commented - not implemented in compared implementation
            # if si < self.val_shape_factor_sd:
            #     self.val_shape_factor_sd = self.val_shape_factor_sd * self.kd
            #     si1 = self.val_shape_factor_sd
            # elif si > self.val_shape_factor_sd:
            #     self.val_shape_factor_sd = self.val_shape_factor_sd / self.kd
            #     si1 = self.val_shape_factor_sd
        else:
            si = -1 * np.log(var_gene) * self.shaping_scaling_factor_fs
            si1 = si
            si2 = si

        # commented - not implemented in compared implementation
        # if best_gene < mean_gene:
        #     si2 = si * self.asymmetry_factor_af
        # elif best_gene > mean_gene:
        #     si1 = si * self.asymmetry_factor_af

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
