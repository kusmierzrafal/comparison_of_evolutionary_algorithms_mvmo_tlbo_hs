from algorithms_comparison.tlbo_comparison.another_tlbo import (
    fitness_levy,
    fitness_rosenbrock,
    fitness_zakharov,
)
from optimization_functions.optimization_functions import (
    levy_function,
    rosenbrock_function,
    zakharov_function,
)

RESULT_FILE = "tlbo_comparison.log"
SEED = 42
POPULATIONS = 30
POPULATIONS_SIZES = [10, 40, 70]
DIMENSIONS = 6

POP_SIZE_ITERATIONS = {10: 20000, 40: 5000, 70: 2857}

TYPICAL_BOUNDARIES = (-10, 10)

LEVY_BOUNDARIES = ROSENBROCK_BOUNDARIES = ZAKHAROV_BOUNDARIES = TYPICAL_BOUNDARIES

COMPARISON_FUNCTIONS = [levy_function, rosenbrock_function, zakharov_function]
COMPARISON_FUNCTIONS_ANOTHER = [fitness_levy, fitness_rosenbrock, fitness_zakharov]

COMPARISON_BOUNDARIES = [LEVY_BOUNDARIES, ROSENBROCK_BOUNDARIES, ZAKHAROV_BOUNDARIES]
