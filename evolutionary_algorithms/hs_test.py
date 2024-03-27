from evolutionary_algorithms.hs import HS
from optimization_functions.optimization_functions import zakharov_function


def test_reproduction():
    dimensions = 6
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, dimensions, boundaries, 0.9)

    population = optimizer.init_population(5)
    child = optimizer.reproduction(population)
    assert len(child) == dimensions
    assert all(boundaries[0] <= gene <= boundaries[1] for gene in child)


def test_evaluation():
    boundaries = (-5.12, 5.12)
    optimizer = HS(10000, 6, boundaries, 0.9)
    population = optimizer.init_population(5)
    evaluated_population = optimizer.evaluation(population, zakharov_function)

    assert len(evaluated_population) == len(population)
