import copy

from numpy.random import seed

from algorithms_comparison.tlbo_comparison.another_tlbo import (
    fitness_levy,
    fitness_rosenbrock,
    fitness_zakharov,
)
from evolutionary_algorithms.evolutionary.population import Population
from optimization_functions.optimization_functions import (
    levy_function,
    rosenbrock_function,
    zakharov_function,
)

RESULT_FILE = "tlbo_comparison.log"
SEED = 42
POPULATIONS = 30
POPULATIONS_SIZES = [10, 40, 70]
DIMENSIONS = [10, 20]

POP_SIZE_ITERATIONS = {10: 500, 40: 500, 70: 500}

TYPICAL_BOUNDARIES = (-10, 10)

LEVY_BOUNDARIES = ROSENBROCK_BOUNDARIES = ZAKHAROV_BOUNDARIES = TYPICAL_BOUNDARIES

COMPARISON_FUNCTIONS = [levy_function, rosenbrock_function, zakharov_function]
COMPARISON_FUNCTIONS_ANOTHER = [fitness_levy, fitness_rosenbrock, fitness_zakharov]

COMPARISON_BOUNDARIES = [LEVY_BOUNDARIES, ROSENBROCK_BOUNDARIES, ZAKHAROV_BOUNDARIES]

seed(SEED)
COMPARISON_POPULATIONS = {
    key: {key: [] for key in POPULATIONS_SIZES} for key in DIMENSIONS
}
COMPARISON_POPULATIONS_ANOTHER = {
    key: {key: [] for key in POPULATIONS_SIZES} for key in DIMENSIONS
}

for dim in DIMENSIONS:
    for pop_size in POPULATIONS_SIZES:
        for _ in range(POPULATIONS):
            population = Population(dim, pop_size, TYPICAL_BOUNDARIES)
            population_another = [ind for ind in copy.deepcopy(population.population.T)]
            COMPARISON_POPULATIONS[dim][pop_size].append(population)
            COMPARISON_POPULATIONS_ANOTHER[dim][pop_size].append(population_another)
