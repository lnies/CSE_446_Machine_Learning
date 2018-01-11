# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:22:40 2018

@author: Lukas
Lukas
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

# Create a class for storing information of the person i 
class person(object):
    index = 0
    age = 0
    salary = 0
    degree = 0
# Create numpy array of i persons
persons = np.ndarray((10,),dtype=np.object)
# Initialize array of persons
for i in range(10):
    persons[i] = person()
# Fill data into persons array    
for i in range(10):
    persons[i].index = i+1
    persons[i].age = data[0][i]
    persons[i].salary = data[1][i]
    persons[i].degree = data[2][i]
# Command to delete i'th element of persons array
#persons = np.delete(persons,1,0)
# Function to print current list of persons 
def print_persons(persons):
    for i in range(len(persons)):
        print("Person ", persons[i].index, ", Age: ", persons[i].age, ", Salary: ", persons[i].salary, ", Degree: ", persons[i].degree)
    return 0
# Boolean variable for choosing method of aclculating error
method = 0 # Standard error 
# Function for calculating the mutual information
# If one of the denominators is 0 just set the whole term to 0
def mutual_inf(n0, p0, n1, p1, D):
    mut_inf = 0
    if ( (n0+p0)*(n0+n1) != 0 ):
        mut_inf += (n0/D) * np.log10( (D*n0)/((n0+p0)*(n0+n1)) )
    elif ( (n0+p0)*(p0+p1) != 0 ):
        mut_inf += (p0/D) * np.log10( (D*p0)/((n0+p0)*(p0+p1)) )
    elif ( (n1+p1)*(n0+n1) != 0 ):
        mut_inf += (n1/D) * np.log10( (D*n1)/((n1+p1)*(n0+n1)) )
    elif ( (n1+p1)*(p0+p1) != 0 ):
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
def calc_error(persons, j_max, steps, feature, method, title):
    n0 = 0 # Number of people with age lower TH and no college degree
    p0 = 0 # Number of people with age lower TH and college degree
    n1 = 0 # Number of people with age higher TH and no college degree
    p1 = 0 # Number of people with age higher TH and college degree
    # Error value storage array
    error = []
    for j in range(0,j_max,steps):
        # Threshold
        TH = j
        for i in range(len(persons)):
            if ( feature == 0 ): 
                if (persons[i].age < TH ): #Hard condition
                    if (persons[i].degree == 0):
                        n0 += 1
                    else: 
                        p0 += 1
                else:
                    if (persons[i].degree == 0):
                        n1 += 1
                    else: 
                        p1 += 1
            if ( feature == 1 ): 
                if (persons[i].salary < TH ): #Hard condition
                    if (persons[i].degree == 0):
                        n0 += 1
                    else: 
                        p0 += 1
                else:
                    if (persons[i].degree == 0):
                        n1 += 1
                    else: 
                        p1 += 1
        # Calculate error for threshold TH
        if method == 0:
            # Error is calculated by the non-majority answer in the subset 
            if ( n0 > p0 ):
                err += p0
            else:
                err += n0
             if ( n0 > p0 ):
                err += p0
            else:
                err += n0 
            error.append([j,n1+p0])
        elif method == 1:
            error.append([j,mutual_inf(n0,p0,n1,p1,10)])
        else:
            print("WRONG PARAMETER FOR CHOOSING METHOD!")
            break
        # Reset counters
        n0 = 0 # Number of people with age lower TH and no college degree
        p0 = 0 # Number of people with age lower TH and college degree
        n1 = 0 # Number of people with age higher TH and no college degree
        p1 = 0 # Number of people with age higher TH and college degree
    # Plot result
    
    plt.figure()
    plt.plot(*zip(*error))
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Error")
    plt.grid(True)
    minimum = 10000
    for i in range(len(error)):
        if ( error[i][1] < minimum ):
            minimum = error[i][1]
            label = i 
        else:
            continue 
    print("Minimum: ", minimum, " at TH: ", label)
        
    


calc_error(persons,120000,1,1,0,"Age as feature (with normal error estimation)")



print_persons(persons)

# Built greedy tree
def DTreeTrain():
    
    return 0










            
            
            




