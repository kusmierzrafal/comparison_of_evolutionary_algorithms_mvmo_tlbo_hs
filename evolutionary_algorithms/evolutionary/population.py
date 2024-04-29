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

    def evaluate_denormalized_ind(self, child: np.ndarray, fitness_function: callable):
        child = child * (self.boundaries[1] - self.boundaries[0]) + self.boundaries[0]
        return fitness_function(child)

    def sort(self):
        sort = np.argsort(self.evaluations)
        self.population = self.population[:, sort]
        self.evaluations = self.evaluations[sort]

    def get_mean_individual(self):
        return np.mean(self.population, axis=1)

    def get_var_individual(self):
        return np.var(self.population, axis=1)

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

    def get_size(self):
        return self.size

    def get_dimensions(self):
        return self.dimensions

    def get_boundaries(self):
        return self.boundaries

    def update_best_population(self, n_best, child: np.ndarray, child_val: float):
        cur_pop_size = self.population.shape[-1]
        if cur_pop_size < n_best:
            self.append_child(child, child_val)
        else:
            self.update_population(child, child_val)

    def append_child(self, child: np.ndarray, child_val: float):
        self.population = np.append(self.population, child, axis=1)
        self.evaluations = np.append(self.evaluations, child_val)

    def archive_best_population(self, n_best, optimize_function: callable):
        self.evaluate(optimize_function)
        self.sort()
        self.population = self.population[:, :n_best]
        self.evaluations = self.evaluations[:n_best]

    def update_population(self, child: np.ndarray, child_val: float):
        worst_ind = np.argmax(self.evaluations)

        if self.evaluations[worst_ind] > child_val:
            self.evaluations[worst_ind] = child_val
            self.population.T[worst_ind] = child.flatten()

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
