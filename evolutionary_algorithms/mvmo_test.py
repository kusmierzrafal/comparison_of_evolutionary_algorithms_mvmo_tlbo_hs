import random

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.population import Population


def test_general_mvmo():
    zakharov_opt_val = 0
    pop_size = 30
    iterations = int(200000 / pop_size)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-5.12, 5.12)
    optimizer = MVMO(1, 10, 1, 3, 75)
    population = Population(dimensions, pop_size, boundaries)
    best_val = optimizer.optimize(
        population, iterations, zakharov_function, zakharov_opt_val
    )
    assert best_val == 5.6354105746494535e-09


def general_mvmo_cec():
    zakharov_cec_function = cec2022_func(func_num=1).values
    zakharov_opt_val = 300
    pop_size = 30
    iterations = int(1000000 / pop_size)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-100, 100)
    optimizer = MVMO(1, 10, 1, 3, 75)
    population = Population(dimensions, pop_size, boundaries)
    optimizer.optimize(population, iterations, zakharov_cec_function, zakharov_opt_val)


if __name__ == "__main__":
    general_mvmo_cec()
