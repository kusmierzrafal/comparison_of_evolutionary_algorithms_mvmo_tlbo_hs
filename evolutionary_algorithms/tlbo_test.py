import numpy as np

from evolutionary_algorithms.tlbo import TLBO
from optimization_functions.optimization_functions import zakharov_function


def test_mutation():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1, dimensions, boundaries)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(
        population, zakharov_function
    )
    mutated_population = optimizer.mutation(
        population, best_individual[0], mean_individual
    )

    assert len(mutated_population) == len(population)
    assert len(mutated_population[0]) == len(population[0]) == dimensions


def test_evaluation():
    dimensions = 2
    boundaries = (-5.12, 5.12)

    def sphere_function(x):
        return x[0] ** 2 + x[1] ** 2

    optimizer = TLBO(1, dimensions, boundaries)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(
        population, sphere_function
    )

    assert len(best_individual) == dimensions
    assert len(mean_individual) == dimensions
    assert all(
        sphere_function(best_individual[0]) <= ind[1] for ind in evaluated_population
    )

    optimizer = TLBO(1, dimensions, boundaries)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(
        population, zakharov_function
    )

    assert len(best_individual) == dimensions
    assert len(mean_individual) == dimensions
    assert all(
        zakharov_function(best_individual[0]) <= ind[1]
        for ind in evaluated_population
    )


def test_crossover():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1, dimensions, boundaries)
    population = optimizer.init_population(5)
    evaluated_population, best_individual, mean_individual = optimizer.evaluation(
        population, zakharov_function
    )
    crossed_population = optimizer.crossover(evaluated_population)
    assert len(crossed_population) == len(population)
    assert len(crossed_population[0]) == len(population[0]) == dimensions


def test_ensure_boundaries_individual():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = TLBO(1, dimensions, boundaries)

    individual_over_boundaries = np.asarray([-10, 10, -10, 10, -10, 10])

    checked_individual = optimizer.ensure_boundaries_individual(
        individual_over_boundaries
    )

    assert np.array_equal(
        checked_individual, np.asarray([-5.12, 5.12, -5.12, 5.12, -5.12, 5.12])
    )
