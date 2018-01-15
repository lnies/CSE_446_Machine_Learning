# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:58:42 2018

@author: Lukas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###### Read data for calculation

data = pd.read_csv('data_2.txt', sep=' ', header=None)

# data[column: 0->2][row: 0->9]
## Column: 0 is [Age / yrs]
## Column: 1 is [Salary / $] 
## Column: 2 is [College Degree (0: no, 1:yes)] 

##### Build tree for feature "age"

# Boolean variable for choosing method of aclculating error
method = 0 # Standard error 
# Function for calculating the mutual information
# If one of the denominators is 0 just set the whole term to 0
def mutual_inf(n0, p0, n1, p1, D):
    mut_inf = 0
    if ( ((n0+p0)*(n0+n1)) != 0 and (D*n0)/((n0+p0)*(n0+n1)) != 0 ):
        mut_inf += (n0/D) * np.log10( (D*n0)/((n0+p0)*(n0+n1)) )
    elif ( (n0+p0)*(p0+p1) != 0 and (D*p0)/((n0+p0)*(p0+p1)) != 0 ):
        mut_inf += (p0/D) * np.log10( (D*p0)/((n0+p0)*(p0+p1)) )
    elif ( (n1+p1)*(n0+n1) != 0 and (D*n1)/((n1+p1)*(n0+n1)) != 0 ):
        mut_inf += (n1/D) * np.log10( (D*n1)/((n1+p1)*(n0+n1)) )
    elif ( (n1+p1)*(p0+p1) != 0 and (D*p1)/((n1+p1)*(p0+p1)) != 0 ):
        mut_inf += (p1/D) * np.log10( (D*p1)/((n1+p1)*(p0+p1)) )
    return mut_inf
# Function for determining the Sign of a multivariante split
#def multisplit(age, income):
#    return np.sign( alpha*age + beta*income -1 )
# Function for calculating the error and printing the graph
alpha = 1
beta = 1
alpha_arr = []
beta_arr = []
tupel = []
def calc_error(alpha_max, steps, method, title):
    n0 = 0 # Number of people with age lower TH and no college degree
    p0 = 0 # Number of people with age lower TH and college degree
    n1 = 0 # Number of people with age higher TH and no college degree
    p1 = 0 # Number of people with age higher TH and college degree
    # Error value storage array
    error = []
    for beta in range(0, 100, 1):
        beta /= 1
        print(beta)
        for alpha in range(0,alpha_max,steps):
            # Threshold
            alpha /= 1
            for i in range(10):
                if ( np.sign( alpha * data[0][i] + beta * data[1][i] -1 ) == -1 ): #Hard condition
                    if (data[2][i] == 0):
                        n0 += 1
                    else: 
                        p0 += 1
                elif ( np.sign( alpha * data[0][i] + beta * data[1][i] -1 ) == 1 ):
                    if (data[2][i] == 0):
                        n1 += 1
                    else: 
                        p1 += 1
                else:
                    print("VALUE ERROR")
            # Calculate error for threshold TH
            if method == 0:
                #error.append([j,n1+p0])
                error.append(n1+p0)
                alpha_arr.append(alpha)
                beta_arr.append(beta)
                tupel.append([alpha, n1+p0])
            elif method == 1:
                error.append([alpha,beta,mutual_inf(n0,p0,n1,p1,10)])
            else:
                print("WRONG PARAMETER FOR CHOOSING METHOD!")
                break
            # Reset counters
            n0 = 0 # Number of people with age lower TH and no college degree
            p0 = 0 # Number of people with age lower TH and college degree
            n1 = 0 # Number of people with age higher TH and no college degree
            p1 = 0 # Number of people with age higher TH and college degree

    


calc_error(10,1,0,"Age as feature (with normal error estimation)")












            
            
            


