import copy
import logging
import random
import time

import numpy as np
from compare_mvmo_consts import (
    AF,
    COMPARISON_BOUNDARIES,
    COMPARISON_FUNCTIONS,
    COMPARISON_LIST_BOUNDARIES,
    COMPARISON_POPULATIONS,
    COMPARISON_POPULATIONS_ANOTHER,
    DIMENSIONS,
    FS,
    MUTATION_SIZE,
    N_BEST_SIZE,
    POP_SIZE_ITERATIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    RESULT_FILE,
    SD,
    SEED,
)
from tqdm import tqdm

from algorithms_comparison.mvmo_comparison.MVMO import MVMO as another_MVMO
from evolutionary_algorithms.mvmo import MVMO

logging.basicConfig(filename=RESULT_FILE, filemode="w", format="%(message)s")

np.random.seed(SEED)
random.seed(SEED)


for function, boundaries, list_boundaries in zip(
    COMPARISON_FUNCTIONS, COMPARISON_BOUNDARIES, COMPARISON_LIST_BOUNDARIES
):

    comparison_populations = copy.deepcopy(COMPARISON_POPULATIONS)
    comparison_populations_another = copy.deepcopy(COMPARISON_POPULATIONS_ANOTHER)

    for dim in DIMENSIONS:
        print(f"function -> {function.__name__}")
        for pop_size in POPULATIONS_SIZES:
            print(f"pop_size -> {pop_size}")

            my_results = []
            other_results = []
            my_times = []
            other_times = []

            for i in tqdm(range(POPULATIONS)):

                population = comparison_populations[dim][pop_size][i]
                population_another = comparison_populations_another[dim][pop_size][i]

                np.random.seed(SEED)
                random.seed(SEED)
                start = time.time()
                optimizer = MVMO(MUTATION_SIZE, N_BEST_SIZE, FS, AF, SD)
                best_val = optimizer.optimize(
                    population, POP_SIZE_ITERATIONS[pop_size], function
                )
                my_times.append(time.time() - start)
                my_results.append(round(best_val, 8))

                np.random.seed(SEED)
                random.seed(SEED)
                start = time.time()
                optimizer = another_MVMO(
                    logger=False,
                    iterations=POP_SIZE_ITERATIONS[pop_size],
                    num_mutation=MUTATION_SIZE,
                    scaling_factor=FS,
                    population_size=N_BEST_SIZE,
                )
                best_val = function(
                    optimizer.optimize(
                        obj_fun=function,
                        x0=population_another,
                        bounds=list_boundaries[dim],
                    )["x"]
                )
                other_times.append(time.time() - start)
                other_results.append(round(best_val, 8))

            logging.warning(
                f"DIMENSIONS -> {dim}\nPOP_SIZE -> {pop_size}\n"
                f"OPTIMIZATION FUNCTION -> {function.__name__}"
            )
            logging.warning(
                f"my results\n"
                f"\tmean -> {sum(my_results) / len(my_results)}\n"
                f"\tbest -> {min(my_results)}\n"
                f"\tworst -> {max(my_results)}\n"
                f"\t{my_results}"
            )
            logging.warning(
                f"other results\n"
                f"\tmean -> {sum(other_results) / len(other_results)}\n"
                f"\tbest -> {min(other_results)}\n"
                f"\tworst -> {max(other_results)}\n"
                f"\t{other_results}"
            )
            logging.warning(
                f"my times\n"
                f"\tmean -> {sum(my_times) / len(my_times)}\n"
                f"\tbest -> {min(my_times)}\n"
                f"\tworst -> {max(my_times)}\n"
                f"\t{my_times}"
            )
            logging.warning(
                f"other times\n"
                f"\tmean -> {sum(other_times) / len(other_times)}\n"
                f"\tbest -> {min(other_times)}\n"
                f"\tworst -> {max(other_times)}\n"
                f"\t{other_times}"
            )
            logging.warning("-" * 1000)
