import copy
import logging
import random
import time

from compare_tlbo_consts import (
    COMPARISON_BOUNDARIES,
    COMPARISON_FUNCTIONS,
    COMPARISON_FUNCTIONS_ANOTHER,
    COMPARISON_POPULATIONS,
    COMPARISON_POPULATIONS_ANOTHER,
    DIMENSIONS,
    POP_SIZE_ITERATIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    RESULT_FILE,
    SEED,
)
from tqdm import tqdm

from algorithms_comparison.tlbo_comparison.another_tlbo import tlbo
from evolutionary_algorithms.tlbo import TLBO

logging.basicConfig(filename=RESULT_FILE, filemode="w", format="%(message)s")

for function, another_function, boundaries in zip(
    COMPARISON_FUNCTIONS, COMPARISON_FUNCTIONS_ANOTHER, COMPARISON_BOUNDARIES
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

                random.seed(SEED)
                start = time.time()
                optimizer = TLBO()
                best_val = optimizer.optimize(
                    population, POP_SIZE_ITERATIONS[pop_size], function
                )
                my_times.append(time.time() - start)
                my_results.append(round(best_val, 8))

                random.seed(SEED)
                start = time.time()
                best_val = another_function(
                    tlbo(
                        another_function,
                        POP_SIZE_ITERATIONS[pop_size],
                        population_another,
                        boundaries,
                    )
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
