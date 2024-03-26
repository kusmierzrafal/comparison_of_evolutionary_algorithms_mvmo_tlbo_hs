# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:14:04 2022

@author: Abhishek Kumar
"""

import numpy as np
import pickle
from cec import cec2022_func
import random

np.random.seed(42)
random.seed(42)


# left for tests
# x = 200.0*np.random.rand(nx,mx)*0.0-100.0
# nx = 10  # dimensions - 2, 10, 20
# mx = 10  # population size - int
fx_n = 12  # function - 1, 2, 3, ..., 11, 12

CEC = cec2022_func(func_num=fx_n)


# read population
with open('./populations/init_pop_100_nr_1', 'rb') as handle:
        population = pickle.load(handle)

results = CEC.values(population)
print(results)
