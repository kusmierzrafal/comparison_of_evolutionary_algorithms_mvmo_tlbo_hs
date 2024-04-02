import numpy as np

from optimization_functions import optimization_functions


def test_rastrigins_function():

    pop_size = 6

    dimensions_optimum = {
        1: 40.35329,
        2: 80.70658,
        3: 121.05987,
        4: 161.41316,
        5: 201.76645,
        6: 242.11974,
        7: 282.47303,
    }

    for dim in range(1, 8):

        # global maximum for different dimensions tested
        test_population = np.zeros((dim, pop_size)) + 4.52299
        assert all(
            np.round(optimization_functions.rastrigins_function(test_population), 5)
            == dimensions_optimum[dim]
        )


def test_zakharov_function():

    pop_size = 6
    opt_value = 0

    for dim in range(1, 6):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(optimization_functions.zakharov_function(test_population), 2)
            == opt_value
        )


def test_rosenbrock_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size)) + 1
        assert all(
            np.round(optimization_functions.rosenbrock_function(test_population), 2)
            == opt_value
        )


def test_expanded_schaffers_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(
                optimization_functions.expanded_schaffers_function(test_population), 2
            )
            == opt_value
        )


def test_bent_cigar_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))
        assert all(
            np.round(optimization_functions.bent_cigar_function(test_population), 2)
            == opt_value
        )


def test_levy_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size)) + 1
        assert all(
            np.round(optimization_functions.levy_function(test_population), 2)
            == opt_value
        )


def test_high_conditioned_elliptic_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(
                optimization_functions.high_conditioned_elliptic_function(
                    test_population
                ),
                2,
            )
            == opt_value
        )


def test_happycat_function():

    pop_size = 6
    opt_value = 0

    for dim in range(1, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size)) - 1

        assert all(
            np.round(optimization_functions.happycat_function(test_population), 2)
            == opt_value
        )


def test_discus_function():

    pop_size = 6
    opt_value = 0

    for dim in range(1, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(optimization_functions.discus_function(test_population), 2)
            == opt_value
        )


def test_ackleys_function():

    pop_size = 6
    opt_value = 0

    for dim in range(1, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(optimization_functions.ackleys_function(test_population), 2)
            == opt_value
        )


def test_schaffers_f7_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(optimization_functions.schaffers_f7_function(test_population), 2)
            == opt_value
        )


def test_hgbat_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size)) - 1

        assert all(
            np.round(optimization_functions.hgbat_function(test_population), 2)
            == opt_value
        )


def test_griewanks_function():

    pop_size = 6
    opt_value = 0

    for dim in range(2, 7):

        # global minimum for different dimensions tested
        test_population = np.zeros((dim, pop_size))

        assert all(
            np.round(optimization_functions.griewanks_function(test_population), 2)
            == opt_value
        )
