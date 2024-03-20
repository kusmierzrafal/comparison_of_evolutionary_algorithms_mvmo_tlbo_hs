import logging
import pickle
import random

import numpy as np
from algorithms_comparison.tlbo_comparison.another_tlbo import tlbo
from evolutionary_algorithms.tlbo import TLBO
from tqdm import tqdm

from compare_tlbo_consts import (
    COMPARISON_BOUNDARIES,
    COMPARISON_FUNCTIONS,
    COMPARISON_FUNCTIONS_ANOTHER,
    DIMENSIONS,
    POP_SIZE_ITERATIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    RESULT_FILE,
    SEED,
)

logging.basicConfig(filename=RESULT_FILE, filemode="w", format="%(message)s")

np.random.seed(SEED)
random.seed(SEED)


for function, another_function, boundaries in zip(
    COMPARISON_FUNCTIONS, COMPARISON_FUNCTIONS_ANOTHER, COMPARISON_BOUNDARIES
):
    print(f"function -> {function.__name__}")
    for pop_size in POPULATIONS_SIZES:
        print(f"pop_size -> {pop_size}")

        my_results = []
        other_results = []

        for i in tqdm(range(POPULATIONS)):
            with open(f"./populations/init_pop_{pop_size}_nr_{i+1}", "rb") as handle:
                population = pickle.load(handle)

            my_results.append(
                function(
                    TLBO(
                        POP_SIZE_ITERATIONS[pop_size], DIMENSIONS, boundaries, False
                    ).optimize(population, function)
                )
            )
            other_results.append(
                another_function(
                    tlbo(
                        another_function,
                        POP_SIZE_ITERATIONS[pop_size],
                        population,
                        boundaries,
                    )
                )
            )

        logging.warning(
            f"POP_SIZE -> {pop_size}\n" f"OPTIMIZATION FUNCTION -> {function.__name__}"
        )
        logging.warning(
            f"my results\n"
            f"\tmean -> {sum(my_results) / len(my_results)}\n"
            f"\tbest -> {min(my_results)}\n"
            f"\t{my_results}"
        )
        logging.warning(
            f"other results\n"
            f"\tmean -> {sum(other_results) / len(other_results)}\n"
            f"\tbest -> {min(other_results)}\n"
            f"\t{other_results}"
        )
        logging.warning("-" * 1000)
