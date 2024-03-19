import logging
import pickle
import random

import numpy as np
from evolutionary_algorithms.hs import HS
from tqdm import tqdm

from compare_hs_consts import (
    COMPARISON_BOUNDARIES,
    COMPARISON_FUNCTIONS,
    DIMENSIONS,
    HMCR,
    MAXIMIZE,
    NUM_PROCESSES,
    POP_SIZE_ITERATIONS,
    POPULATIONS,
    POPULATIONS_SIZES,
    RESULT_FILE,
    SEED,
    WORKERS,
)
from objective_function_levy import ObjectiveFunction as ObjectiveLevy
from objective_function_rosenbrock import ObjectiveFunction as ObjectiveRosenbrock
from objective_function_zakharov import ObjectiveFunction as ObjectiveZakharov
from pyharmonysearch import harmony_search

COMPARISON_OBJECTIVE_FUNCTIONS = [ObjectiveLevy, ObjectiveRosenbrock, ObjectiveZakharov]

logging.basicConfig(filename=RESULT_FILE, filemode="a", format="%(message)s")

np.random.seed(SEED)
random.seed(SEED)

if __name__ == "__main__":
    for function, obj_function, boundaries in zip(
        COMPARISON_FUNCTIONS, COMPARISON_OBJECTIVE_FUNCTIONS, COMPARISON_BOUNDARIES
    ):
        print(f"function -> {function.__name__}")
        for pop_size in POPULATIONS_SIZES:
            print(f"pop_size -> {pop_size}")

            my_results = []
            other_results = []

            obj_fun = obj_function(
                iterations=POP_SIZE_ITERATIONS[pop_size], population_size=pop_size
            )

            for i in tqdm(range(POPULATIONS)):
                with open(
                    f"./populations/init_pop_{pop_size}_nr_{i+1}", "rb"
                ) as handle:
                    population = pickle.load(handle)

                population_list = [ind.tolist() for ind in population]

                my_results.append(
                    HS(
                        POP_SIZE_ITERATIONS[pop_size],
                        DIMENSIONS,
                        boundaries,
                        MAXIMIZE,
                        hmcr=HMCR,
                    ).optimize(population, function)[1]
                )

                other_results.append(
                    harmony_search(
                        obj_fun,
                        NUM_PROCESSES,
                        WORKERS,
                        initial_harmonies=population_list,
                    ).best_fitness
                )

            logging.warning(
                f"POP_SIZE -> {pop_size}\n"
                f"OPTIMIZATION FUNCTION -> {function.__name__}"
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
