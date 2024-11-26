import os

files_to_check = os.listdir('./tests/hyperparameters_tuning/')
for file in files_to_check:
    if file.endswith('txt'):
        f = open(f'./hyperparameters_tuning/{file}')
        content = f.read().rstrip().split('\n')
        f.close()
        if len(content) != 16 and '.' in content[-1]:
            os.remove(f'./tests/hyperparameters_tuning/{file}')

files_to_check = os.listdir('./tests/final_test/mvmo/')

mvmo_done = []
tlbo_done = []
hs_done = []


for file in files_to_check:
    if file.startswith('mvmo'):
        mvmo_done.append(file)
    elif file.startswith('tlbo'):
        tlbo_done.append(file)
    elif file.startswith('hs'):
        hs_done.append(file)



mvmo_done = [(int(mvmo.split('_')[-5]), int(mvmo.split('_')[-2]), int(mvmo.split('_')[-1].split('.')[-2])) for mvmo in mvmo_done]
tlbo_done = [(int(tlbo.split('_')[-5]), int(tlbo.split('_')[-2]), int(tlbo.split('_')[-1].split('.')[-2])) for tlbo in tlbo_done]
hs_done = [(int(hs.split('_')[-5]), int(hs.split('_')[-2]), int(hs.split('_')[-1].split('.')[-2])) for hs in hs_done]

mvmo_all = []
for run in range(1, 31, 1):
    for dim in [20, 10]:
        for func in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            mvmo_all.append((func, dim, run))

mvmo_to_do = [mvmo for mvmo in mvmo_all if mvmo not in mvmo_done]


tlbo_all = []

for run in range(1, 31, 1):
    for dim in [20, 10]:
        for func in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            tlbo_all.append((func, dim, run))

tlbo_to_do = [tlbo for tlbo in tlbo_all if tlbo not in tlbo_done]

hs_all = []
for run in range(1, 31, 1):
    for dim in [20, 10]:
        for func in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            hs_all.append((func, dim, run))

hs_to_do = [hs for hs in hs_all if hs not in hs_done]

# import pdb; pdb.set_trace()

i = -1
for func, dim, run in mvmo_to_do:
    i += 1
    if i % 1 == 0:
        print('\n\n\n')
        print('#' * 50)
        print(f"""
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/library/
import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population



OPT_VAL = {{
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
}}

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
        seeds = list(seeds.replace('\\t', '').replace('\\r', '').replace(' ', '').split('\\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{{'_'.join(str(v) for v in parameters.values())}}__{{str(func_num)}}_{{str(pop_size)}}_{{str(iterations)}}_{{str(dim)}}_{{str(run)}}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo
 
        """)
    print(f"""
dim = {dim}
iterations = 200000 if dim == 10 else 1000000
pop_size = 50
func_num = {func}
run = {run}

parameters={{
        "optimizer": 'mvmo',
        "mutation_size": 1,
        "n_best_size": 40,
        "shaping_scaling_factor_fs": 1,
        "asymmetry_factor_af": 1,
        "val_shape_factor_sd": 75
}}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)
""")



i = -1
for func, dim, run in tlbo_to_do:
    i += 1
    if i % 1 == 0:
        print('\n\n\n')
        print('#' * 50)
        print(f"""
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/library/
import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population



OPT_VAL = {{
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
}}

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
        seeds = list(seeds.replace('\\t', '').replace('\\r', '').replace(' ', '').split('\\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{{'_'.join(str(v) for v in parameters.values())}}__{{str(func_num)}}_{{str(pop_size)}}_{{str(iterations)}}_{{str(dim)}}_{{str(run)}}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo

        """)
    print(f"""
dim = {dim}
pop_size = 60
iterations = 1666 if dim == 10 else 8333
func_num = {func}
run = {run}

parameters={{
    "optimizer": 'tlbo',
    }}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)

""")

i = -1
for func, dim, run in hs_to_do:
    i += 1
    if i % 1 == 0:
        print('\n\n\n')
        print('#' * 50)
        print(f"""
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/library/
import random
import logging

import numpy as np
from optimization_functions.cec.CEC2022 import cec2022_func
from optimization_functions.optimization_functions import zakharov_function

from evolutionary_algorithms.mvmo import MVMO
from evolutionary_algorithms.hs import HS
from evolutionary_algorithms.tlbo import TLBO
from evolutionary_algorithms.evolutionary.population import Population



OPT_VAL = {{
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
}}

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
        seeds = list(seeds.replace('\\t', '').replace('\\r', '').replace(' ', '').split('\\n'))[:-1]
        seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]

    np.random.seed(seeds[int(seed_ind)])
    random.seed(seeds[int(seed_ind)])

    result_file = f"{{'_'.join(str(v) for v in parameters.values())}}__{{str(func_num)}}_{{str(pop_size)}}_{{str(iterations)}}_{{str(dim)}}_{{str(run)}}.txt"
    logging.warning(result_file)

    optimizer = get_optimizer(optimizer, parameters)
    population = Population(dim, pop_size, BOUNDARIES)
    optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)


# iterations = 200000 hs mvmo
# iterations = 100000 # tlbo

        """)
    print(f"""
dim = {dim}
iterations = 200000 if dim == 10 else 1000000
pop_size = 50
func_num = {func}
run = {run}

parameters={{
        "optimizer": 'hs',
        "pcr": 0.93,
        "mutation_size": 0.25,
        "mutation_factor": 0.18,
}}

run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)
""")
