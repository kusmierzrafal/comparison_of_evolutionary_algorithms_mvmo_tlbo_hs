from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 1

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 4

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 7

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 10

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 13

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 16

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 19

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 22

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 25

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 28

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 2,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 1

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 4

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 7

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 10

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 13

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 16

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 19

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 22

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 25

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 28

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 10,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 1

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 4

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 7

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 10

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 13

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 16

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 19

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 22

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 25

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 28

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 20,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 1

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 4

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 7

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 10

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 13

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 16

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 19

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 22

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 25

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 28

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 30,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 1

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 4

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 7

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 10

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 13

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 16

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 19

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 22

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 25

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


from google.colab import drive

drive.mount('/content/drive')
%cd / content / drive / MyDrive / library /

import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population

OPT_VAL = {
    1: 300,
    2: 400,
    3: 600,
    4: 800,
    5: 900,
    6: 1800,
    7: 2000,
    8: 2200,
    9: 2300,
    10: 2400,
    11: 2600,
    12: 2700
}

BOUNDARIES = (-100, 100)
RUNS = 30


def get_optimizer(optimizer, parameters):
    if optimizer == 'mvmo':
        return MVMO(
            parameters["mutation_size"],
            parameters["n_best_size"],
            parameters["shaping_scaling_factor_fs"],
            parameters["asymmetry_factor_af"],
            parameters["val_shape_factor_sd"]
        )
    elif optimizer == 'hs':
        return HS(
            parameters["pcr"],
            parameters["mutation_size"],
            parameters["mutation_factor"]
        )
    elif optimizer == 'tlbo':
        return TLBO(
        )


def run_test(parameters, func_num, pop_size, iterations, dim, run):
    optimizer = parameters["optimizer"]
    cec_function = cec2022_func(func_num=func_num).values

    seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
    seed_ind = seed_ind % 1000 + 1

    with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
        seeds = handle.read()
        seeds = list(seeds.replace('\t', '').replace('\r', '').replace(' ', '').split('\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{'_'.join(str(v) for v in parameters.values())}__{str(func_num)}_{str(pop_size)}_{str(iterations)}_{str(dim)}_{str(run)}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo


iterations = 200000
dim = 20
pop_size = 50
func_num = 6
run = 28

parameters = {
    "optimizer": 'mvmo',
    "mutation_size": 1,
    "n_best_size": 40,
    "shaping_scaling_factor_fs": 1,
    "asymmetry_factor_af": 1,
    "val_shape_factor_sd": 75
}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 1)

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run + 2)

##################################################


