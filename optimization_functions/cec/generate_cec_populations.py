import numpy as np
from evolutionary_algorithms.mvmo import MVMO
import pickle
import random

np.random.seed(42)
random.seed(42)

rosenbrock_boundaries = (-10, 10)
dimensions = 10
pop_size_num = {
    10: 10,
    100: 10,
    1000: 10
}

# prepare populations
mvmo_optimizer = MVMO(100_000, dimensions, rosenbrock_boundaries, False, mutation_size=1)
for pop_size in [10, 100, 1000]:
    for i in range(pop_size_num[pop_size]):
        population = mvmo_optimizer.init_population(pop_size)
        with open(f'./populations/init_pop_{pop_size}_nr_{i+1}', 'wb') as handle:
            pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)

# read population
# with open('./mvmo_populations/init_pop_10_nr_1', 'rb') as handle:
#         mvmo_population = pickle.load(handle)
