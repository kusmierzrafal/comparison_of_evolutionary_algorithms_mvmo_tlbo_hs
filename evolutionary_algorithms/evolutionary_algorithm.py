import numpy as np


class EvolutionaryAlgorithm:

    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
    ):
        self.iterations = iterations
        self.dimensions = dimensions
        self.boundaries = boundaries

    def init_population(self, size: int = 2) -> np.ndarray[np.ndarray]:
        """
        Initialize population of given size with individuals of given dimension and constraints
        :param size: size of initialized population
        :return: numpy array of dimensions (numpy arrays with population size length)
        """
        return np.random.uniform(
            low=self.boundaries[0],
            high=self.boundaries[1],
            size=(self.dimensions, size),
        )
