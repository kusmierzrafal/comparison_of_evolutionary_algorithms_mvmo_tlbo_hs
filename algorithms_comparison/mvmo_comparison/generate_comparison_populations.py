import numpy as np
from evolutionary_algorithms.mvmo import MVMO
import pickle
import random

np.random.seed(42)
random.seed(42)

rosenbrock_boundaries = (-10, 10)
dimensions = 6
pop_sizes = [10, 40, 70]
number_of_populations = 30

# prepare populations for comparison with another implementation
mvmo_optimizer = MVMO(100_000, dimensions, rosenbrock_boundaries, False, mutation_size=1)
for pop_size in pop_sizes:
    for i in range(number_of_populations):
        population = mvmo_optimizer.init_population(pop_size)
        with open(f'./populations/init_pop_{pop_size}_nr_{i+1}', 'wb') as handle:
            pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)

# read population
# with open('./populations/init_pop_10_nr_1', 'rb') as handle:
#         mvmo_population = pickle.load(handle)
