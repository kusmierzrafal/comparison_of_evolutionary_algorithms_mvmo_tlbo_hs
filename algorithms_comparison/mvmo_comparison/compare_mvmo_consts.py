import copy

from numpy.random import seed

from evolutionary_algorithms.evolutionary.population import Population
from optimization_functions.optimization_functions import (
    levy_function,
    rosenbrock_function,
    zakharov_function,
)

RESULT_FILE = "mvmo_comparison.log"
SEED = 42
POPULATIONS = 30
POPULATIONS_SIZES = [10, 40, 70]
DIMENSIONS = [10, 20]
MUTATION_SIZE = 3
N_BEST_SIZE = 10
FS = 1
AF = 1
SD = 75

POP_SIZE_ITERATIONS = {10: 500, 40: 500, 70: 500}

TYPICAL_BOUNDARIES = (-10, 10)

TYPICAL_LIST_BOUNDARIES = {}

for dim in DIMENSIONS:
    TYPICAL_LIST_BOUNDARIES[dim] = [
        (TYPICAL_BOUNDARIES[0], TYPICAL_BOUNDARIES[1]) for _ in range(dim)
    ]

LEVY_BOUNDARIES = ROSENBROCK_BOUNDARIES = ZAKHAROV_BOUNDARIES = TYPICAL_BOUNDARIES
LEVY_LIST_BOUNDARIES = ROSENBROCK_LIST_BOUNDARIES = ZAKHAROV_LIST_BOUNDARIES = (
    TYPICAL_LIST_BOUNDARIES
)

COMPARISON_FUNCTIONS = [levy_function, rosenbrock_function, zakharov_function]
COMPARISON_BOUNDARIES = [LEVY_BOUNDARIES, ROSENBROCK_BOUNDARIES, ZAKHAROV_BOUNDARIES]
COMPARISON_LIST_BOUNDARIES = [
    LEVY_LIST_BOUNDARIES,
    ROSENBROCK_LIST_BOUNDARIES,
    ZAKHAROV_LIST_BOUNDARIES,
]

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
            population_another = copy.deepcopy(population.get_normalized().T)
            COMPARISON_POPULATIONS[dim][pop_size].append(population)
            COMPARISON_POPULATIONS_ANOTHER[dim][pop_size].append(population_another)
