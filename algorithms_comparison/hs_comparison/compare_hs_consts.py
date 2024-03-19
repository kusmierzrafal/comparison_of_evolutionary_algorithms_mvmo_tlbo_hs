from optimization_functions.optimization_functions import (
    levy_function,
    rosenbrock_function,
    zakharov_function,
)

RESULT_FILE = "hs_comparison.log"
SEED = 42
POPULATIONS = 30
POPULATIONS_SIZES = [10, 40, 70]
DIMENSIONS = 6
NUM_PROCESSES = 1
WORKERS = 1

POP_SIZE_ITERATIONS = {10: 2, 40: 2, 70: 2}

TYPICAL_BOUNDARIES = (-10, 10)

LEVY_BOUNDARIES = ROSENBROCK_BOUNDARIES = ZAKHAROV_BOUNDARIES = TYPICAL_BOUNDARIES

COMPARISON_FUNCTIONS = [levy_function, rosenbrock_function, zakharov_function]
COMPARISON_BOUNDARIES = [LEVY_BOUNDARIES, ROSENBROCK_BOUNDARIES, ZAKHAROV_BOUNDARIES]

# another implementation consts
TYPICAL_LOWER_BOUNDS = [TYPICAL_BOUNDARIES[0] for _ in range(DIMENSIONS)]
TYPICAL_UPPER_BOUNDS = [TYPICAL_BOUNDARIES[1] for _ in range(DIMENSIONS)]
VARIABLE = [True] * DIMENSIONS
MAXIMIZE = False
HMCR = 0.93
PAR_OFF = 0
