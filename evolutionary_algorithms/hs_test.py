import random

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.evolutionary.population import Population


def test_general_hs():
    zakharov_opt_val = 0
    pop_size = 30
    iterations = int(200000)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-5.12, 5.12)
    population = Population(dimensions, pop_size, boundaries)
    optimizer = HS(0.9, 0.25, 0.2)
    best_val = optimizer.optimize(
        population, iterations, zakharov_function, zakharov_opt_val, "hs_test.txt"
    )

    assert best_val == 1.0191034289361441e-05


def general_hs_cec():
    zakharov_cec_function = cec2022_func(func_num=1).values
    zakharov_opt_val = 300
    pop_size = 30
    iterations = int(200000)
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dimensions = 10
    boundaries = (-100, 100)
    optimizer = HS(0.9, 0.25, 0.2)
    population = Population(dimensions, pop_size, boundaries)
    optimizer.optimize(population, iterations, zakharov_cec_function, zakharov_opt_val, "hs_test.txt")


if __name__ == "__main__":
    general_hs_cec()
