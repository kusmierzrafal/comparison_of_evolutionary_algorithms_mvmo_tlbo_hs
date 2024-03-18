import pickle
import random

import numpy as np
from evolutionary_algorithms.hs import HS

from compare_hs_consts import (
    DIMENSIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    SEED,
    TYPICAL_BOUNDARIES,
)

np.random.seed(SEED)
random.seed(SEED)

# prepare populations for comparison with another implementation
# iterations don't make difference in this case
hs_optimizer = HS(100_000, DIMENSIONS, TYPICAL_BOUNDARIES, False, hmcr=1)
for pop_size in POPULATIONS_SIZES:
    for i in range(POPULATIONS):
        population = hs_optimizer.init_population(pop_size)
        with open(f"./populations/init_pop_{pop_size}_nr_{i+1}", "wb") as handle:
            pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)

# example of reading population
# with open('./populations/init_pop_10_nr_1', 'rb') as handle:
#         hs_population = pickle.load(handle)
