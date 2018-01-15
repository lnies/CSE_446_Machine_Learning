# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:05:09 2018

@author: Lukas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function of the error reduction rate
def error_red(n0, p0, n1, p1, D):
    return ( 1/D ) * ( np.minimum(n0+n1,p0+p1) - ( np.minimum(n0,p0) + np.minimum(n1,p1) ) )

# Function for calculating the mutual information
# If one of the denominators is 0 just set the whole term to 0
def mutual_inf(n0, p0, n1, p1, D):
    #print(n0,p0,n1,p1)
    mut_inf = 0
    if ( ((n0+p0)*(n0+n1)) != 0 and (D*n0)/((n0+p0)*(n0+n1)) != 0 ):
        mut_inf += (n0/D) * np.log10( (D*n0)/((n0+p0)*(n0+n1)) )
    elif ( (n0+p0)*(p0+p1) != 0 and (D*p0)/((n0+p0)*(p0+p1)) != 0 ):
        mut_inf += (p0/D) * np.log10( (D*p0)/((n0+p0)*(p0+p1)) )
    elif ( (n1+p1)*(n0+n1) != 0 and (D*n1)/((n1+p1)*(n0+n1)) != 0 ):
        mut_inf += (n1/D) * np.log10( (D*n1)/((n1+p1)*(n0+n1)) )
    elif ( (n1+p1)*(p0+p1) != 0 and (D*p1)/((n1+p1)*(p0+p1)) != 0 ):
        mut_inf += (p1/D) * np.log10( (D*p1)/((n1+p1)*(p0+p1)) )
    return( mut_inf )
    
D = 1000
mut_inf = np.ndarray(0)
err_red = np.ndarray(0)
indeces = np.ndarray(0)
for i in range(1,500):
    n0 = i
    p1 = i
    n1 = 500 - i
    p0 = 500 - i
    indeces = np.append(indeces,i)
    mut_inf = np.append(mut_inf,mutual_inf(n0,p0,n1,p1,D))
    err_red = np.append(err_red,error_red(n0,p0,n1,p1,D))
    
plt.figure()
plt.plot(indeces,mut_inf,label="Mutual information")
plt.plot(indeces,err_red,label="Error reduction rate")
plt.legend()
plt.title("Mutual information")
plt.xlabel("Index i")
plt.ylabel("")
plt.grid(True)
    

























