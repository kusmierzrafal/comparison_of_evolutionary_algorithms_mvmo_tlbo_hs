# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:14:04 2022

@author: Abhishek Kumar
"""

import numpy as np
from CEC2022 import cec2022_func

nx = 10     # dimensions - 2, 10, 20
mx = 7     # population size - int
fx_n = 12   # function - 1, 2, 3, ..., 11, 12

CEC = cec2022_func(func_num = fx_n)

x = 200.0*np.random.rand(nx,mx)*0.0-100.0
result = CEC.values(x)
print(result)