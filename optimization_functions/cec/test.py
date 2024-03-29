# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:14:04 2022

@author: Abhishek Kumar
"""

import numpy as np
from CEC2022 import cec2022_func

nx = 10
mx = 10
fx_n = 12

CEC = cec2022_func(func_num = fx_n)

x = 200.0*np.random.rand(nx,mx)*0.0-100.0
F = CEC.values(x)
print(F.ObjFunc)