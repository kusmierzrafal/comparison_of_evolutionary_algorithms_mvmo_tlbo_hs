import logging
import pickle
import random

import numpy as np
from algorithms_comparison.mvmo_comparison.MVMO import MVMO as another_MVMO
from evolutionary_algorithms.mvmo import MVMO

from compare_mvmo_consts import (
    COMPARISON_BOUNDARIES,
    COMPARISON_FUNCTIONS,
    COMPARISON_LIST_BOUNDARIES,
    DIMENSIONS,
    MUTATION_SIZE,
    POP_SIZE_ITERATIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    RESULT_FILE,
    SEED,
)

logging.basicConfig(filename=RESULT_FILE, filemode="w", format="%(message)s")

np.random.seed(SEED)
random.seed(SEED)


for function, boundaries, list_boundaries in zip(
    COMPARISON_FUNCTIONS, COMPARISON_BOUNDARIES, COMPARISON_LIST_BOUNDARIES
):
    print(f"function -> {function.__name__}")
    for pop_size in POPULATIONS_SIZES:
        print(f"pop_size -> {pop_size}")

        my_results = []
        other_results = []

        for i in range(POPULATIONS):
            with open(f"./populations/init_pop_{pop_size}_nr_{i+1}", "rb") as handle:
                population = pickle.load(handle)

            normalized_population = [
                (ind - boundaries[0]) / (boundaries[1] - boundaries[0])
                for ind in population
            ]

            my_results.append(
                MVMO(
                    POP_SIZE_ITERATIONS[pop_size],
                    DIMENSIONS,
                    boundaries,
                    False,
                    mutation_size=MUTATION_SIZE,
                ).optimize(population, function)[1]
            )
            other_results.append(
                function(
                    another_MVMO(
                        logger=False,
                        iterations=POP_SIZE_ITERATIONS[pop_size],
                        num_mutation=MUTATION_SIZE,
                        population_size=pop_size,
                    ).optimize(
                        obj_fun=function,
                        x0=np.asarray(normalized_population),
                        bounds=list_boundaries,
                    )[
                        "x"
                    ]
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
