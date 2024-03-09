import logging
from random import choice, randint, random

import numpy as np

from evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization_functions.optimization_functions import rastrigins_function


class TLBO(EvolutionaryAlgorithm):
    def __init__(
        self,
        iterations: int,
        dimensions: int,
        boundaries: tuple[float, float],
        maximize: bool,
    ):
        """
        Teaching Learning Based Optimization Algorithm
        :param iterations: number of iterations during optimization
        :type iterations: int
        :param dimensions: number of dimensions of optimization function
        :type dimensions: int
        :param boundaries: lower and higher limit of the range of every gene
        :type boundaries: tuple of floats
        :param maximize: True for maximization, False for minimization
        :type maximize: bool
        """
        self.teaching_factor = randint(1, 2)

        logging.basicConfig(filename="tlbo.log", filemode="a", format="%(message)s")

        super().__init__(iterations, dimensions, boundaries, maximize)

        if self.maximize:
            self.get_better = self.get_better_max
        elif not self.maximize:
            self.get_better = self.get_better_min

    @staticmethod
    def get_better_max(new_population, old_population):
        better_population = [
            new_ind if new_ind[1] > ind[1] else ind
            for new_ind, ind in zip(new_population, old_population)
        ]
        return better_population

    @staticmethod
    def get_better_min(new_population, old_population):
        better_population = [
            new_ind if new_ind[1] < ind[1] else ind
            for new_ind, ind in zip(new_population, old_population)
        ]
        return better_population

    def optimize(self, population: list[np.ndarray], optimize_function: callable):
        """
        Searches for the best solution for a given number of iterations
        :param population: initial population
        :type population: list[np.ndarray]
        :param optimize_function:
        :type optimize_function: callable
        :return: best from found solutions
        :rtype: numpy.ndarray
        """

        evaluated_population, best_individual, mean_individual = self.evaluation(
            population, optimize_function
        )

        for i in range(self.iterations):

            mutated_population = self.mutation(
                [ind[0] for ind in evaluated_population],
                best_individual[0],
                mean_individual,
            )
            evaluated_mutated_population = self.evaluation(
                mutated_population, optimize_function
            )[0]
            evaluated_mutated_population = self.get_better(
                evaluated_mutated_population, evaluated_population
            )

            crossed_population = self.crossover(evaluated_mutated_population)
            evaluated_crossed_population = self.evaluation(
                crossed_population, optimize_function
            )[0]
            evaluated_crossed_population = self.get_better(
                evaluated_mutated_population, evaluated_crossed_population
            )

            best_individual = sorted(
                evaluated_crossed_population,
                key=lambda ind: ind[1],
                reverse=self.maximize,
            )[0]
            evaluated_population = evaluated_crossed_population
            mean_individual = np.mean([ind[0] for ind in evaluated_population], axis=0)

        return best_individual[0]

    def mutation(
        self,
        population: list[np.ndarray],
        best_individual: np.ndarray,
        mean_individual: np.ndarray,
    ):
        """
        Mutates every individual in the population
        :param population: population to be mutated
        :type population: list[numpy.ndarray]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: evaluated mutated population
        :rtype: list[tuple[numpy.ndarray, float]]
        """
        random_factor = np.array([random() for _ in range(len(mean_individual))])
        mutagen = random_factor * (
            best_individual - self.teaching_factor * mean_individual
        )
        mutated_population = [ind + mutagen for ind in population]
        mutated_population = self.ensure_boundaries_population(mutated_population)

        return mutated_population

    def evaluation(self, population: list[np.ndarray], fitness_function: callable):
        """
        Counts fitness function value for every individual
        :param population: population to be evaluated
        :type population: list[numpy.ndarray]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: evaluated population, best individual, mean individual
        :rtype: tuple[list[tuple[numpy.ndarray, float]], numpy.ndarray, numpy.ndarray]
        """
        evaluated_population = [(ind, fitness_function(ind)) for ind in population]
        best_individual = sorted(
            evaluated_population, key=lambda ind: ind[1], reverse=self.maximize
        )[0]
        mean_individual = np.mean(population, axis=0)

        return evaluated_population, best_individual, mean_individual

    def ensure_boundaries_individual(self, new_ind: np.ndarray) -> np.ndarray:
        """
        Sets every gene value to lower/higher boundary value if it crosses the given range
        :param new_ind: individual to be validated
        :type new_ind: numpy.ndarray
        :return: validated individual
        :rtype: numpy.ndarray
        """
        return np.array(
            [
                (
                    self.boundaries[0]
                    if gene < self.boundaries[0]
                    else (self.boundaries[1] if gene > self.boundaries[1] else gene)
                )
                for gene in new_ind
            ]
        )

    def ensure_boundaries_population(
        self, new_pop: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        For the entire population sets every gene value to lower/higher boundary value if it crosses the given range
        :param new_pop: population which individuals are to validation
        :type new_pop: list[numpy.ndarray]
        :return: validated population
        :rtype: list[numpy.ndarray]
        """
        return [self.ensure_boundaries_individual(new_ind) for new_ind in new_pop]

    def crossover(
        self, evaluated_population: list[tuple[np.ndarray, float]]
    ) -> list[np.ndarray]:
        """
        For every individual draws other individual with other fitness function value and crosses them
        :param evaluated_population: evaluated population
        :type evaluated_population: list[tuple[numpy.ndarray, float]]
        :param fitness_function: function to evaluate how close a given solution is to the optimum solution
        :type fitness_function: callable
        :return: population after crossover
        :rtype: list[numpy.ndarray]
        """
        crossed_population: list[np.ndarray] = []
        r = random()
        for ind in evaluated_population:
            to_choose = list(
                filter(lambda individual: individual[1] != ind[1], evaluated_population)
            )
            if len(to_choose) == 0:
                return [individual[0] for individual in evaluated_population]
            else:
                ind_to_cross = choice(to_choose)

            if self.maximize:
                if ind[1] > ind_to_cross[1]:
                    new_ind = np.array(
                        [g1 + r * (g1 - g2) for g1, g2 in zip(ind[0], ind_to_cross[0])]
                    )
                else:
                    new_ind = np.array(
                        [g1 + r * (g2 - g1) for g1, g2 in zip(ind[0], ind_to_cross[0])]
                    )
            else:
                if ind[1] < ind_to_cross[1]:
                    new_ind = np.array(
                        [g1 + r * (g1 - g2) for g1, g2 in zip(ind[0], ind_to_cross[0])]
                    )
                else:
                    new_ind = np.array(
                        [g1 + r * (g2 - g1) for g1, g2 in zip(ind[0], ind_to_cross[0])]
                    )

            new_ind = self.ensure_boundaries_individual(new_ind)

            crossed_population.append(new_ind)
        return crossed_population


if __name__ == "__main__":
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1000, 6, boundaries, True)
    population = optimizer.init_population(50)
    optimizer.optimize(population, rastrigins_function)
