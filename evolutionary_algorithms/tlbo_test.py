import random

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.population import Population
from evolutionary_algorithms.tlbo import TLBO


def test_general_tlbo():
    zakharov_opt_val = 0
    pop_size = 30
    iterations = int(200000 / pop_size)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-5.12, 5.12)
    optimizer = TLBO()
    population = Population(dimensions, pop_size, boundaries)
    best_val = optimizer.optimize(
        population, iterations, zakharov_function, zakharov_opt_val
    )
    assert best_val == 8.626902861448757e-09


def general_tlbo_cec():
    zakharov_cec_function = cec2022_func(func_num=1).values
    zakharov_opt_val = 300
    pop_size = 30
    iterations = int(1000000 / pop_size)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-100, 100)
    optimizer = TLBO()
    population = Population(dimensions, pop_size, boundaries)
    optimizer.optimize(population, iterations, zakharov_cec_function, zakharov_opt_val)


if __name__ == "__main__":
    general_tlbo_cec()
