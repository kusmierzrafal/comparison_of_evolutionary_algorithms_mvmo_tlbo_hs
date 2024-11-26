import os



# files_to_check = os.listdir('./hyperparameters_tuning/')
# for file in files_to_check:
#     if file.endswith('txt'):
#         f = open(f'./hyperparameters_tuning/{file}')
#         content = f.read()
#         length = len(content.rstrip().split('\n'))
#         if length != 16:
#             print(file)


#
files_to_check = os.listdir('./hyperparameters_tuning/verified/')
mvmo_done = []
tlbo_done = []

for file in files_to_check:
    if file.startswith('mvmo'):
        mvmo_done.append(file)
    elif file.startswith('tlbo'):
        tlbo_done.append(file)


mvmo_done = [(int(mvmo.split('_')[2]), int(mvmo.split('_')[-1].split('.')[0])) for mvmo in mvmo_done]
tlbo_done = [(int(tlbo.split('_')[3]), int(tlbo.split('_')[-1].split('.')[0])) for tlbo in tlbo_done]

mvmo_all = []
for best_size in [2, 10, 20, 30, 40]:
    for run in range(1, 31, 1):
        mvmo_all.append((best_size, run))

mvmo_to_do = [mvmo for mvmo in mvmo_all if mvmo not in mvmo_done]

tlbo_all = []
for pop_size in range(10, 80, 10):
    for run in range(1, 31, 1):
        tlbo_all.append((pop_size, run))

tlbo_to_do = [tlbo for tlbo in tlbo_all if tlbo not in tlbo_done]
import pdb; pdb.set_trace()
#
# for best_size, run in mvmo_to_do:
#
#
#     print(f"""
#         from google.colab import drive
#         drive.mount('/content/drive')
#         %cd /content/drive/MyDrive/library/
#         import random
#         import logging
#
#         import numpy as np
#         from optimization_functions.cec.CEC2022 import cec2022_func
#         from optimization_functions.optimization_functions import zakharov_function
#
#         from evolutionary_algorithms.mvmo import MVMO
#         from evolutionary_algorithms.hs import HS
#         from evolutionary_algorithms.tlbo import TLBO
#         from evolutionary_algorithms.evolutionary.population import Population
#
#
#
#         OPT_VAL = {{
#             1: 300,
#             2: 400,
#             3: 600,
#             4: 800,
#             5: 900,
#             6: 1800,
#             7: 2000,
#             8: 2200,
#             9: 2300,
#             10: 2400,
#             11: 2600,
#             12: 2700
#         }}
#
#         BOUNDARIES = (-100, 100)
#         RUNS = 30
#
#         def get_optimizer(optimizer, parameters):
#             if optimizer == 'mvmo':
#                 return MVMO(
#                     parameters["mutation_size"],
#                     parameters["n_best_size"],
#                     parameters["shaping_scaling_factor_fs"],
#                     parameters["asymmetry_factor_af"],
#                     parameters["val_shape_factor_sd"]
#                     )
#             elif optimizer == 'hs':
#                 return HS(
#                     parameters["pcr"],
#                     parameters["mutation_size"],
#                     parameters["mutation_factor"]
#                   )
#             elif optimizer == 'tlbo':
#                 return TLBO(
#                   )
#
#         def run_test(parameters, func_num, pop_size, iterations, dim, run):
#
#             optimizer = parameters["optimizer"]
#             cec_function = cec2022_func(func_num=func_num).values
#
#
#             seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
#             seed_ind = seed_ind % 1000 + 1
#
#             with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
#                 seeds = handle.read()
#                 seeds = list(seeds.replace('\\t', '').replace('\\r', '').replace(' ', '').split('\\n'))[:-1]
#                 seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]
#
#             np.random.seed(seeds[int(seed_ind)])
#             random.seed(seeds[int(seed_ind)])
#
#             result_file = f"{{'_'.join(str(v) for v in parameters.values())}}__{{str(func_num)}}_{{str(pop_size)}}_{{str(iterations)}}_{{str(dim)}}_{{str(run)}}.txt"
#             logging.warning(result_file)
#
#             optimizer = get_optimizer(optimizer, parameters)
#             population = Population(dim, pop_size, BOUNDARIES)
#             optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)
#
#
#         # iterations = 200000 hs mvmo
#         # iterations = 100000 # tlbo
#
#         iterations = 200000
#         dim = 20
#         pop_size = 50
#         func_num = 6
#         run = {run}
#
#         parameters={{
#                 "optimizer": 'mvmo',
#                 "mutation_size": 1,
#                 "n_best_size": {best_size},
#                 "shaping_scaling_factor_fs": 1,
#                 "asymmetry_factor_af": 1,
#                 "val_shape_factor_sd": 75
#         }}
#
#         run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)
#
# """)
#
#     print('\n\n\n')
#     print('#' * 50)
#
#
#
# for pop_size, run in tlbo_to_do:
#
#     print(f"""
#         from google.colab import drive
#         drive.mount('/content/drive')
#         %cd /content/drive/MyDrive/library/
#         import random
#         import logging
#
#         import numpy as np
#         from optimization_functions.cec.CEC2022 import cec2022_func
#         from optimization_functions.optimization_functions import zakharov_function
#
#         from evolutionary_algorithms.mvmo import MVMO
#         from evolutionary_algorithms.hs import HS
#         from evolutionary_algorithms.tlbo import TLBO
#         from evolutionary_algorithms.evolutionary.population import Population
#
#
#
#         OPT_VAL = {{
#             1: 300,
#             2: 400,
#             3: 600,
#             4: 800,
#             5: 900,
#             6: 1800,
#             7: 2000,
#             8: 2200,
#             9: 2300,
#             10: 2400,
#             11: 2600,
#             12: 2700
#         }}
#
#         BOUNDARIES = (-100, 100)
#         RUNS = 30
#
#         def get_optimizer(optimizer, parameters):
#             if optimizer == 'mvmo':
#                 return MVMO(
#                     parameters["mutation_size"],
#                     parameters["n_best_size"],
#                     parameters["shaping_scaling_factor_fs"],
#                     parameters["asymmetry_factor_af"],
#                     parameters["val_shape_factor_sd"]
#                     )
#             elif optimizer == 'hs':
#                 return HS(
#                     parameters["pcr"],
#                     parameters["mutation_size"],
#                     parameters["mutation_factor"]
#                   )
#             elif optimizer == 'tlbo':
#                 return TLBO(
#                   )
#
#         def run_test(parameters, func_num, pop_size, iterations, dim, run):
#
#             optimizer = parameters["optimizer"]
#             cec_function = cec2022_func(func_num=func_num).values
#
#
#             seed_ind = (dim / 10 * func_num * RUNS + run) - RUNS
#             seed_ind = seed_ind % 1000 + 1
#
#             with open('./optimization_functions/cec/input_data/Rand_Seeds.txt', 'r') as handle:
#                 seeds = handle.read()
#                 seeds = list(seeds.replace('\\t', '').replace('\\r', '').replace(' ', '').split('\\n'))[:-1]
#                 seeds = [int(float(seed[:4]) * 10 ** int(seed[-1:])) for seed in seeds]
#
#             np.random.seed(seeds[int(seed_ind)])
#             random.seed(seeds[int(seed_ind)])
#
#             result_file = f"{{'_'.join(str(v) for v in parameters.values())}}__{{str(func_num)}}_{{str(pop_size)}}_{{str(iterations)}}_{{str(dim)}}_{{str(run)}}.txt"
#             logging.warning(result_file)
#
#             optimizer = get_optimizer(optimizer, parameters)
#             population = Population(dim, pop_size, BOUNDARIES)
#             optimizer.optimize(population, iterations, cec_function, OPT_VAL[func_num], result_file)
#
#
#         # iterations = 200000 hs mvmo
#         # iterations = 100000 # tlbo
#
#         dim = 20
#         pop_size = {pop_size}
#         iterations = int(200000 / 2 // {pop_size})
#         func_num = 6
#         run = {run}
#
#         parameters={{
#             "optimizer": 'tlbo',
#             }}
#
#         run_test(parameters=parameters, func_num=func_num, pop_size=pop_size, iterations=iterations, dim=dim, run=run)
#
# """)
#
#     print('\n\n\n')
#     print('#' * 50)
#
