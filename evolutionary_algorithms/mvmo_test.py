import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from evolutionary_algorithms.mvmo import MVMO
from optimization_functions.optimization_functions import rastrigins_function


def test_init_population():
    optimizer = MVMO(10000, 3, (-5.12, 5.12), True, 2)
    population_1 = optimizer.normalize_population(optimizer.init_population(size=3))
    population_2 = optimizer.init_population(10)

    assert len(population_1) == 3
    assert len(population_1[0]) == 3
    assert 0 <= population_1[0][0] <= 1
    assert 0 <= population_1[1][1] <= 1
    assert 0 <= population_1[2][2] <= 1

    assert len(population_2) == 10
    assert len(population_2[0]) == 3
    assert -5.12 <= population_2[0][0] <= 5.12
    assert -5.12 <= population_2[1][1] <= 5.12
    assert -5.12 <= population_2[2][2] <= 5.12


def test_de_normalize_population():
    optimizer = MVMO(10000, 5, (-5.12, 5.12), True, 3)
    population = optimizer.init_population(5)
    normalized_population = optimizer.normalize_population(population)
    denormalized_population = optimizer.denormalize_population(normalized_population)

    assert np.array_equal(denormalized_population, population)
    assert max([ind.max() for ind in normalized_population]) <= 1.0
    assert min([ind.min() for ind in normalized_population]) >= 0.0


def test_transformation():

    x = [num / 100 for num in range(101)]

    # Effects of mean of dynamic population on the transformation function h
    y = [MVMO.transformation(random_xi, 0, si1=10, si2=10) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.25, si1=10, si2=10) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y3 = [MVMO.transformation(random_xi, 0.75, si1=10, si2=10) for random_xi in x]
    y4 = [MVMO.transformation(random_xi, 1.0, si1=10, si2=10) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)
    assert all(0 <= v <= 1 for v in y3)
    assert all(0 <= v <= 1 for v in y4)

    # Effects of shaping scaling factor on the transformation function h
    y = [MVMO.transformation(random_xi, 0.5, si1=0, si2=0) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.5, si1=5, si2=5) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y3 = [MVMO.transformation(random_xi, 0.5, si1=15, si2=15) for random_xi in x]
    y4 = [MVMO.transformation(random_xi, 0.5, si1=50, si2=50) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)
    assert all(0 <= v <= 1 for v in y3)
    assert all(0 <= v <= 1 for v in y4)

    # Effects of different shape factors si1 =/= si2
    y = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=20) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=20, si2=10) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)


def test_count_si():
    optimizer = MVMO(10000, 5, (-5.12, 5.12), True, 3)
    last_no_zero_si = 20
    assert optimizer.count_si(0.5, 0.5, np.nan, last_no_zero_si)[0] == last_no_zero_si
    assert optimizer.count_si(0.5, 0.5, np.inf, last_no_zero_si)[0] == last_no_zero_si


def test_mutation():
    optimizer = MVMO(1000, 6, (-5.12, 5.12), True, 2)
    population = optimizer.init_population(4)
    normalized_population = optimizer.normalize_population(population)
    best_population, mean_individual, var_individual = optimizer.evaluation(
        normalized_population, rastrigins_function
    )
    best_individual = best_population[0][0]
    mutated_population = optimizer.mutation(
        normalized_population, mean_individual, var_individual, best_individual
    )

    assert all(
        mutated == best
        for (mutated, best) in list(zip(mutated_population[0], best_individual))[2:]
    )
    assert all(
        mutated == best
        for (mutated, best) in list(zip(mutated_population[1], best_individual))[:2]
    )
    assert all(
        mutated == best
        for (mutated, best) in list(zip(mutated_population[1], best_individual))[4:]
    )
    assert all(
        mutated == best
        for (mutated, best) in list(zip(mutated_population[2], best_individual))[:4]
    )
    assert all(
        mutated == best
        for (mutated, best) in list(zip(mutated_population[3], best_individual))[2:]
    )

    assert all(0.0 <= gene <= 1.0 for gene in mutated_population[0][:2])
    assert all(0.0 <= gene <= 1.0 for gene in mutated_population[1][2:4])
    assert all(0.0 <= gene <= 1.0 for gene in mutated_population[2][4:])
    assert all(0.0 <= gene <= 1.0 for gene in mutated_population[3][:2])


def test_evaluation():
    optimizer = MVMO(1000, 6, (-5.12, 5.12), True, 3)
    population = optimizer.init_population(5)

    best_population, mean_individual, var_individual = optimizer.evaluation(
        population, rastrigins_function
    )
    assert len(best_population) == 2
    assert len(best_population[0][0]) == len(mean_individual) == len(var_individual)


def test_plot_transformation():
    # plt.show() commented
    mpl.use("TkAgg")

    x = [num / 100 for num in range(101)]

    # Effects of mean of dynamic population on the transformation function h
    y = [MVMO.transformation(random_xi, 0, si1=10, si2=10) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.25, si1=10, si2=10) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y3 = [MVMO.transformation(random_xi, 0.75, si1=10, si2=10) for random_xi in x]
    y4 = [MVMO.transformation(random_xi, 1.0, si1=10, si2=10) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)
    assert all(0 <= v <= 1 for v in y3)
    assert all(0 <= v <= 1 for v in y4)

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    ax.plot(x, y, "blue", label="x` = 0")
    ax.plot(x, y1, "red", label="x` = 0.25")
    ax.plot(x, y2, "green", label="x` = 0.5")
    ax.plot(x, y3, "purple", label="x` = 0.75")
    ax.plot(x, y4, "cyan", label="x` = 1")
    ax.legend()
    plt.ylabel("x")
    plt.xlabel("x'")
    plt.title("Wpływ średniej wartości genu na funkcję mapującą\ndla s = 10")
    # plt.show()

    # Effects of shaping scaling factor on the transformation function h
    y = [MVMO.transformation(random_xi, 0.5, si1=0, si2=0) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.5, si1=5, si2=5) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y3 = [MVMO.transformation(random_xi, 0.5, si1=15, si2=15) for random_xi in x]
    y4 = [MVMO.transformation(random_xi, 0.5, si1=50, si2=50) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)
    assert all(0 <= v <= 1 for v in y3)
    assert all(0 <= v <= 1 for v in y4)

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)

    ax.plot(x, y, "blue", label="s = 0")
    ax.plot(x, y1, "red", label="s = 5")
    ax.plot(x, y2, "green", label="s = 10")
    ax.plot(x, y3, "purple", label="s = 15")
    ax.plot(x, y4, "cyan", label="s = 50")
    ax.legend()
    plt.ylabel("x")
    plt.xlabel("x'")
    plt.title("Wpływ współczynnika skalującego na funkcję mapującą\ndla x` = 0.5")
    # plt.show()

    # Effects of different shape factors si1 =/= si2
    y = [MVMO.transformation(random_xi, 0.5, si1=10, si2=10) for random_xi in x]
    y1 = [MVMO.transformation(random_xi, 0.5, si1=10, si2=20) for random_xi in x]
    y2 = [MVMO.transformation(random_xi, 0.5, si1=20, si2=10) for random_xi in x]

    assert all(0 <= v <= 1 for v in y)
    assert all(0 <= v <= 1 for v in y1)
    assert all(0 <= v <= 1 for v in y2)

    fig, ax = plt.subplots()
    ax.plot(x, y, "green", label="si1 = si2 = 10", linestyle="--")
    ax.plot(x, y1, "red", label="si1 = 10, si2 = 20")

    ax.legend()
    plt.ylabel("xi")
    plt.xlabel("random x")
    plt.title("effects of different shape factors si1 =/= si2 \nfor mean xi = 0.5")
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y, "green", label="si1 = si2 = 10", linestyle="--")
    ax.plot(x, y2, "red", label="si1 = 20, si2 = 10")

    ax.legend()
    plt.ylabel("xi")
    plt.xlabel("random x")
    plt.title("effects of different shape factors si1 =/= si2 \nfor mean xi = 0.5")
    # plt.show()
