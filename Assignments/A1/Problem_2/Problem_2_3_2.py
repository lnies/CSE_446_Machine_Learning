# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:08:21 2018

@author: Lukas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_2.txt', sep=' ', header=None)

alpha = 0.999999617
beta = - 8.75*10**(-4)

for i in range(len(data)):
    res = alpha*data[0][i] + beta*data[1][i] -1
    print(res)