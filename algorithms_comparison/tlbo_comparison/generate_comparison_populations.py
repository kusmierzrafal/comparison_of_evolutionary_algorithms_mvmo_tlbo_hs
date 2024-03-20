import pickle
import random

import numpy as np
from evolutionary_algorithms.tlbo import TLBO

from compare_tlbo_consts import (
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
tlbo_optimizer = TLBO(100_000, DIMENSIONS, TYPICAL_BOUNDARIES, False)
for pop_size in POPULATIONS_SIZES:
    for i in range(POPULATIONS):
        population = tlbo_optimizer.init_population(pop_size)
        with open(f"./populations/init_pop_{pop_size}_nr_{i+1}", "wb") as handle:
            pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)

# example of reading population
# with open('./populations/init_pop_10_nr_1', 'rb') as handle:
#         mvmo_population = pickle.load(handle)
