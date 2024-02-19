import numpy as np

from optimization_functions import optimization_functions


def test_rastrigins_function():
    # global maximum for different dimensions tested
    assert (
        round(optimization_functions.rastrigins_function(np.array([4.52299])), 5)
        == 40.35329
    )
    assert (
        round(
            optimization_functions.rastrigins_function(np.array([4.52299, 4.52299])), 5
        )
        == 80.70658
    )
    assert (
        round(
            optimization_functions.rastrigins_function(
                np.array([4.52299, 4.52299, 4.52299])
            ),
            5,
        )
        == 121.05987
    )
    assert (
        round(
            optimization_functions.rastrigins_function(
                np.array([4.52299, 4.52299, 4.52299, 4.52299])
            ),
            5,
        )
        == 161.41316
    )
    assert (
        round(
            optimization_functions.rastrigins_function(
                np.array([4.52299, 4.52299, 4.52299, 4.52299, 4.52299])
            ),
            5,
        )
        == 201.76645
    )
    assert (
        round(
            optimization_functions.rastrigins_function(
                np.array([4.52299, 4.52299, 4.52299, 4.52299, 4.52299, 4.52299])
            ),
            5,
        )
        == 242.11974
    )
    assert (
        round(
            optimization_functions.rastrigins_function(
                np.array(
                    [4.52299, 4.52299, 4.52299, 4.52299, 4.52299, 4.52299, 4.52299]
                )
            ),
            5,
        )
        == 282.47303
    )


def test_zakharov_function():
    # global minimum for different dimensions tested
    assert (
        round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.zakharov_function(np.array([0, 0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.zakharov_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.zakharov_function(np.array([0])), 2) == 0


def test_rosenbrock_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1, 1, 1])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1, 1])), 2)
        == 0
    )
    assert (
        round(optimization_functions.rosenbrock_function(np.array([1, 1, 1, 1])), 2)
        == 0
    )
    assert (
        round(optimization_functions.rosenbrock_function(np.array([1, 1, 1])), 2) == 0
    )
    assert round(optimization_functions.rosenbrock_function(np.array([1, 1])), 2) == 0
    assert round(optimization_functions.rosenbrock_function(np.array([1])), 2) == 0


def test_expanded_schaffers_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.expanded_schaffers_function(
                np.array([0, 0, 0, 0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.expanded_schaffers_function(
                np.array([0, 0, 0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.expanded_schaffers_function(np.array([0, 0, 0, 0])),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.expanded_schaffers_function(np.array([0, 0, 0])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.expanded_schaffers_function(np.array([0, 0])), 2)
        == 0
    )


def test_bent_cigar_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0, 0, 0])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.bent_cigar_function(np.array([0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.bent_cigar_function(np.array([0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.bent_cigar_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.bent_cigar_function(np.array([0])), 2) == 0


def test_levy_function():
    # global minimum for different dimensions tested
    assert (
        round(optimization_functions.levy_function(np.array([1, 1, 1, 1, 1, 1])), 2)
        == 0
    )
    assert (
        round(optimization_functions.levy_function(np.array([1, 1, 1, 1, 1])), 2) == 0
    )
    assert round(optimization_functions.levy_function(np.array([1, 1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1, 1])), 2) == 0
    assert round(optimization_functions.levy_function(np.array([1])), 2) == 0


def test_high_conditioned_elliptic_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.high_conditioned_elliptic_function(
                np.array([0, 0, 0, 0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.high_conditioned_elliptic_function(
                np.array([0, 0, 0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.high_conditioned_elliptic_function(
                np.array([0, 0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.high_conditioned_elliptic_function(
                np.array([0, 0, 0])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.high_conditioned_elliptic_function(np.array([0, 0])),
            2,
        )
        == 0
    )


def test_happycat_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.happycat_function(
                np.array([-1, -1, -1, -1, -1, -1])
            ),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.happycat_function(np.array([-1, -1, -1, -1, -1])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.happycat_function(np.array([-1, -1, -1, -1])), 2)
        == 0
    )
    assert (
        round(optimization_functions.happycat_function(np.array([-1, -1, -1])), 2) == 0
    )
    assert round(optimization_functions.happycat_function(np.array([-1, -1])), 2) == 0
    assert round(optimization_functions.happycat_function(np.array([-1])), 2) == 0


def test_discus_function():
    # global minimum for different dimensions tested
    assert (
        round(optimization_functions.discus_function(np.array([0, 0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.discus_function(np.array([0, 0, 0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.discus_function(np.array([0, 0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.discus_function(np.array([0, 0])), 2) == 0


def test_ackleys_function():
    # global minimum for different dimensions tested
    assert (
        round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.ackleys_function(np.array([0, 0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.ackleys_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0, 0])), 2) == 0
    assert round(optimization_functions.ackleys_function(np.array([0])), 2) == 0


def test_schaffers_f7_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0, 0, 0])),
            2,
        )
        == 0
    )
    assert (
        round(
            optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0, 0])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.schaffers_f7_function(np.array([0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.schaffers_f7_function(np.array([0, 0])), 2) == 0


def test_hgbat_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.hgbat_function(np.array([-1, -1, -1, -1, -1, -1])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.hgbat_function(np.array([-1, -1, -1, -1, -1])), 2)
        == 0
    )
    assert (
        round(optimization_functions.hgbat_function(np.array([-1, -1, -1, -1])), 2) == 0
    )
    assert round(optimization_functions.hgbat_function(np.array([-1, -1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1, -1])), 2) == 0
    assert round(optimization_functions.hgbat_function(np.array([-1])), 2) == 0


def test_griewanks_function():
    # global minimum for different dimensions tested
    assert (
        round(
            optimization_functions.griewanks_function(np.array([0, 0, 0, 0, 0, 0])), 2
        )
        == 0
    )
    assert (
        round(optimization_functions.griewanks_function(np.array([0, 0, 0, 0, 0])), 2)
        == 0
    )
    assert (
        round(optimization_functions.griewanks_function(np.array([0, 0, 0, 0])), 2) == 0
    )
    assert round(optimization_functions.griewanks_function(np.array([0, 0, 0])), 2) == 0
    assert round(optimization_functions.griewanks_function(np.array([0, 0])), 2) == 0
