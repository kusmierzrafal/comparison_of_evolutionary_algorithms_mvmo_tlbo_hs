import numpy as np


class EvolutionaryAlgorithm:

    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
        maximize: bool,
    ):
        self.iterations = iterations
        self.dimensions = dimensions
        self.boundaries = boundaries
        self.maximize = maximize

    def init_population(self, size: int = 2) -> list[np.ndarray]:
        """
        Initialize population of given size with individuals of given dimension and constraints
        :param size: size of initialized population
        :return: population (list) of individuals (numpy arrays)
        """
        return [
            np.random.uniform(
                low=self.boundaries[0], high=self.boundaries[1], size=(self.dimensions,)
            )
            for _ in range(size)
        ]
