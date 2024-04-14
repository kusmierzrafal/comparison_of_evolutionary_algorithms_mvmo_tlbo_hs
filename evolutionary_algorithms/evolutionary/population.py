import bisect
import copy

import numpy as np


class Population:

    def __init__(
        self,
        dimensions: int,
        size: int,
        boundaries: tuple[float, float],
    ):
        self.dimensions = dimensions
        self.size = size
        self.boundaries = boundaries
        self.population = self.init_population()
        self.evaluations = None
        self.best_population = np.asarray([])
        self.best_evaluations = None

    def init_population(self) -> np.ndarray[np.ndarray]:
        """
        Initialize population of given size with individuals of given dimension and constraints
        :param size: size of initialized population
        :return: numpy array of dimensions (numpy arrays with population size length)
        """
        return np.random.uniform(
            low=self.boundaries[0],
            high=self.boundaries[1],
            size=(self.dimensions, self.size),
        )

    def normalize(self):
        """
        Normalizes the input population
        :return: normalized population
        """
        self.population = (self.population - self.boundaries[0]) / (
            self.boundaries[1] - self.boundaries[0]
        )

    def denormalize(self):
        """
        Denormalizes the input population
        :return: denormalized population
        """
        self.population = (
            self.population * (self.boundaries[1] - self.boundaries[0])
            + self.boundaries[0]
        )

    def get_normalized(self):
        """
        Normalizes the input population
        :return: normalized population
        """
        return (self.population - self.boundaries[0]) / (
            self.boundaries[1] - self.boundaries[0]
        )

    def get_denormalized(self):
        """
        Denormalizes the input population
        :return: denormalized population
        """
        return (
            self.population * (self.boundaries[1] - self.boundaries[0])
            + self.boundaries[0]
        )

    @staticmethod
    def _evaluate(population: np.ndarray[np.ndarray], fitness_function: callable):
        return fitness_function(population)

    def evaluate(self, fitness_function: callable):
        self.evaluations = self._evaluate(self.population, fitness_function)

    def evaluate_denormalized(self, fitness_function: callable):
        denormalized_population = self.get_denormalized()
        self.evaluations = self._evaluate(denormalized_population, fitness_function)

    def sort(self):
        sort = np.argsort(self.evaluations)
        self.population = self.population[:, sort]
        self.evaluations = self.evaluations[sort]

    def sort_best(self):
        sort = np.argsort(self.best_evaluations)
        self.best_population = self.best_population[:, sort]
        self.best_evaluations = self.best_evaluations[sort]

    def get_mean_individual(self, n_best=None):
        return np.mean(self.population[:, :n_best], axis=1)

    def get_var_individual(self, n_best=None):
        return np.var(self.population[:, :n_best], axis=1)

    def get_best_individual(self):
        """
        :return: best individual
        """
        best_ind = np.argmin(self.evaluations)
        return np.copy(self.population[:, best_ind])

    def get_best_value(self):
        best_ind = np.argmin(self.evaluations)
        return self.evaluations[best_ind]

    def get_mean_archive_individual(self):
        return np.mean(self.population, axis=1)

    def get_var_archive_individual(self):
        return np.var(self.best_population, axis=1)

    def get_best_archive_individual(self):
        return np.copy(self.best_population[:, 0])

    def get_best_archive_value(self):
        return self.best_evaluations[0]

    def get_size(self):
        return self.size

    def get_dimensions(self):
        return self.dimensions

    def get_boundaries(self):
        return self.boundaries

    def archive_best_population(self, n_best):
        cur_best_size = self.best_population.shape[-1]
        if cur_best_size < n_best:
            self.init_best_population(cur_best_size, n_best)
        else:
            self.update_best_population(n_best)

    def update_best_population(self, n_best):
        for index in range(self.size):
            if self.evaluations[index] < self.best_evaluations[n_best - 1]:
                insertion_index = bisect.bisect(
                    self.best_evaluations, self.evaluations[index]
                )
                self.best_evaluations = np.insert(
                    self.best_evaluations, insertion_index, self.evaluations[index]
                )[:n_best]
                self.best_population = np.insert(
                    self.best_population,
                    insertion_index,
                    self.population[:, index],
                    axis=1,
                )[:, :n_best]
            else:
                break

    def update_population(self, child: np.ndarray, child_val: float):
        worst_ind = np.argmax(self.evaluations)

        if self.evaluations[worst_ind] > child_val:
            self.evaluations[worst_ind] = child_val
            self.population.T[worst_ind] = child.flatten()

    def init_best_population(self, cur_best_size, n_best):
        if cur_best_size == 0:
            self.best_population = copy.deepcopy(self.population)[:, :n_best]
            self.best_evaluations = copy.deepcopy(self.evaluations)[:n_best]
        else:
            self.best_population = np.hstack(
                [self.best_population, copy.deepcopy(self.population)]
            )
            self.best_evaluations = np.hstack(
                [self.best_evaluations, copy.deepcopy(self.evaluations)]
            )
            self.sort_best()
            self.best_population = self.best_population[:, :n_best]
            self.best_evaluations = self.best_evaluations[:n_best]

    def ensure_boundaries(self):
        mask_lower = self.population < self.boundaries[0]
        self.population[mask_lower] = self.boundaries[0]
        mask_upper = self.population > self.boundaries[1]
        self.population[mask_upper] = self.boundaries[1]

    def get_better(self, other_population):
        mask = self.evaluations < other_population.evaluations

        self.population = np.where(mask, self.population, other_population.population)
        self.evaluations = np.where(
            mask, self.evaluations, other_population.evaluations
        )
