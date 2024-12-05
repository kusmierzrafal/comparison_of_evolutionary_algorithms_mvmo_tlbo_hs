import random
from math import pow

from algorithms_comparison.hs_comparison.compare_hs_consts import (
    HMCR, MAXIMIZE, PAR, MPAP)

from pyharmonysearch import ObjectiveFunctionInterface


class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is a toy objective function that contains only continuous variables. A random seed is used, so the same result
        will be generated every time.

        Goal:

            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 4 at (0, -1).

        Note that since all variables are continuous, we don't actually need to implement get_index() and get_num_discrete_values().

        Warning: Stochastically solving a linear system is dumb. This is just a toy example.
    """

    def __init__(self, iterations, population_size, kwargs):
        self._lower_bounds = kwargs["typical_lower_bounds"]
        self._upper_bounds = kwargs["typical_upper_bounds"]
        self._variable = kwargs["variable"]

        # define all input parameters
        self._maximize = MAXIMIZE  # do we maximize or minimize?
        self._max_imp = iterations  # maximum number of improvisations
        self._hms = population_size  # harmony memory size
        self._hmcr = HMCR  # harmony memory considering rate
        self._par = PAR  # pitch adjusting rate
        self._mpap = MPAP  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        # self._random_seed = 8675309  # optional random seed for reproducible results

    def get_fitness(self, vector):
        """
            minimize Zakharovs function.
            The minimum is 0 at (0, 0, 0, 0, 0, 0).
        """
        return sum([pow(x, 2) for x in vector]) + pow(sum([0.5 * x for x in vector]), 2) + pow(sum([0.5 * x for x in vector]), 4)

    def get_value(self, i, index=None):
        """
            Values are returned uniformly at random in their entire range. Since both parameters are continuous, index can be ignored.
        """
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # all variables are continuous
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_random_seed(self):
        return self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize
