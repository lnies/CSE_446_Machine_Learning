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
    position = 0
# Create a class to built a node for a tree 
class node(object):
    position = 1
    persons = []
    label = 0
    prevnode = 0
    nextnode0 = 0
    nextnode1 = 0
    split_TH = 0
# Create empty numpy array as an array of node
tree = np.ndarray((1),dtype=np.object)
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
    persons[i].position = 1
# Create root node
root = node()
root.position = 1
root.persons = persons
root.label = -1
root.prevnode = -1
# Insert root node into tree
tree[0] = root
# Function to print current list of persons 
def print_persons(persons):
    for i in range(len(persons)):
        print("Person ", persons[i].index, ", Age: ", persons[i].age, ", Salary: ", persons[i].salary, ", Degree: ", persons[i].degree, " Postition: ", persons[i].position)
    return "++++++++++++++++++++++++++++++++++++++++"
# Function to print the tree
def print_tree(tree):
    for i in range(len(tree)):
        print("++++++++++++++++++++++++++++++++++++++++")
        print("Node number: ", i, ", Position in tree: ", tree[i].position, ", List of persons: ")
        print("Threshold split at: ", tree[i].split_TH)
        print(print_persons(tree[i].persons))
# Boolean variable for choosing method of aclculating error
method = 0 # Standard error 
# Function for calculating the mutual information
# If one of the denominators is 0 just set the whole term to 0
def mutual_inf(n0, p0, n1, p1, D):
    #print(n0,p0,n1,p1)
    mut_inf = 0
    if ( ((n0+p0)*(n0+n1)) != 0 and (D*n0)/((n0+p0)*(n0+n1)) != 0 ):
        mut_inf += (n0/D) * np.log2( (D*n0)/((n0+p0)*(n0+n1)) )
    if ( (n0+p0)*(p0+p1) != 0 and (D*p0)/((n0+p0)*(p0+p1)) != 0 ):
        mut_inf += (p0/D) * np.log2( (D*p0)/((n0+p0)*(p0+p1)) )
    if ( (n1+p1)*(n0+n1) != 0 and (D*n1)/((n1+p1)*(n0+n1)) != 0 ):
        mut_inf += (n1/D) * np.log2( (D*n1)/((n1+p1)*(n0+n1)) )
    if ( (n1+p1)*(p0+p1) != 0 and (D*p1)/((n1+p1)*(p0+p1)) != 0 ):
        mut_inf += (p1/D) * np.log2( (D*p1)/((n1+p1)*(p0+p1)) )
    return( (-1)*mut_inf )
# Function for determining the Sign of a multivariante split
#def multisplit(age, income):
#    return np.sign( alpha*age + beta*income -1 )
# Function for calculating the error and printing the graph
alpha = 1
beta = 1
alpha_arr = []
beta_arr = []
tupel = []
# calc_error takes a list of persons and a feature and calculates the minimal error if the node gets split into two subnodes  
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
        err = 0
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
            if ( n1 > p1 ):
                err += p1
            else:
                err += n1 
            error.append([j,err])
        elif method == 1:
             # Error is calculated by the non-majority answer in the subset 
            if ( n0 > p0 ):
                err += p0
            else:
                err += n0
            if ( n1 > p1 ):
                err += p1
            else:
                err += n1 
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
    
    #plt.figure()
    #plt.plot(*zip(*error))
    #plt.title(title)
    #plt.xlabel("Threshold")
    #plt.ylabel("Error")
    #plt.grid(True)
    # Test if the node has to be split or if you already reached true 0 error
    if ( err == 0  ):
        # -1 encoudes this case and no split will be conducted
        minimum = [-1,10000]
        return minimum
    # Calculate best TH forsplitting
    minimum = [0,10000]
    for i in range(len(error)):
        if ( error[i][1] < minimum[1] ):
            minimum[1] = error[i][1]
            minimum[0] = i
        else:
            continue 
    # Check for the case if no minimum can be found: if so, split at least one person to get a better error eventually
    if ( minimum[0] == 0 ):
        # Choose the right method
        if ( method == 0 ):
            # Search for youngest person
            youngest = 1000
            index_youngest = 0
            for i in range(len(persons)):
                if ( persons[i].age < youngest ):
                    youngest = persons[i].age 
                    index_youngest = i
                else:
                    continue
            # Assign threshold for splitting above youngest person
            minimum[0] = persons[index_youngest].age + 1
        elif ( method == 1 ):
            # Search for youngest person
            best = 1000000
            index_best = 0
            for i in range(len(persons)):
                if ( persons[i].salary < best ):
                    best = persons[i].salary 
                    index_best = i
                else:
                    continue
            # Assign threshold for splitting above youngest person
            minimum[0] = persons[index_best].salary + 1
        else:
            print("YOU CHOSE A WRONG PARAMETER FOR CHOOSING A METHOD")
        return(minimum)
    else:
        return(minimum)
    
#tree = np.append(tree,root,axis=None)
# Command to delete i'th element of persons array
#persons = np.delete(persons,1,0)


# Built greedy tree
# nodeindex is the index of the tree-array of the mothernode 
def DTreeTrain(tree, nodeindex, method):
    # Initialize temporary arrays
    persons0 =  np.ndarray((0),dtype=np.object)
    persons1 =  np.ndarray((0),dtype=np.object)
    # Initialize array of persons
    for i in range(0):
        persons0[i] = person()
        persons1[i] = person()
    # Search for a minimal error by scanning thresholds, choose feature with smallet error
    TH_sal = calc_error(tree[nodeindex].persons,120000,1,1,method,"Salary as feature (with normal error estimation)")
    TH_age = calc_error(tree[nodeindex].persons,100,1,0,method,"Age as feature (with normal error estimation)") 
    # Test if node has to be split or not ( if -1 then don't split)
    if ( TH_sal == -1 or TH_age == -1 ):
        return tree
    # Create two new sub-nodes
    node0 = node()
    node1 = node()
    # Choose the feature according to the smalles error
    if ( TH_sal[1] < TH_age[1] ):
        for i in range(len(tree[nodeindex].persons)):
            # If feature of person i is under threshold, person i goes into left node
            if ( tree[nodeindex].persons[i].salary < TH_sal[0] ):
                persons0 = np.append(persons0, tree[nodeindex].persons[i])
            # If feature of person i is above threshold, person i goes into right node
            else:
                persons1 = np.append(persons1, tree[nodeindex].persons[i])
        # Save split threshold in sub-node
        node0.split_TH = TH_sal[0]
        node1.split_TH = TH_sal[0]
    else:
        for i in range(len(tree[nodeindex].persons)):
            # If feature of person i is under threshold, person i goes into left node
            if ( tree[nodeindex].persons[i].age < TH_age[0] ):
                persons0 = np.append(persons0, tree[nodeindex].persons[i])
            # If feature of person i is above threshold, person i goes into right node
            else:
                persons1 = np.append(persons1, tree[nodeindex].persons[i]) 
        # Save split threshold in sub-node
        node0.split_TH = TH_age[0]
        node1.split_TH = TH_age[0]
    # Assign values and insert the the two new nodes
    node0.position = tree[nodeindex].position * 10 
    node0.persons = persons0
    node0.label = -1
    node0.prevnode = 1
    node0.nextnode0 = -1
    node0.nextnode1 = -1
    node1.position = tree[nodeindex].position * 10 + 1
    node1.persons = persons1
    node1.label = -1
    node1.prevnode = 1
    node1.nextnode0 = -1
    node1.nextnode1 = -1
    tree = np.append(tree,node0,axis=None)
    tree = np.append(tree,node1,axis=None)
    #print("Sub_node_0:")
    #print(print_persons(tree[1].persons))
    #print("Sub_node_1:")
    #print_persons(persons1)
    return tree

# Fill the whole tree
i = 0
while True:
    #print("Length Tree: ", len(tree), ", Index: ", i)
    if ( i == len(tree) ):
        break
    tree = DTreeTrain(tree,i,1)
    # Test if there are any empty nodes, otherwise remove them
    for j in range(len(tree)):
        if ( tree[j-1].split_TH == -1 ):
            tree = np.delete(tree,j-1,axis=None)
        else:
            continue
    # Test if there are any node left to split by checking if the index overcounts the length of the tree
    # since the length of tree doesn't change after it's completetly filled 
    if ( i > len(tree) ):
        break
    else:
        i += 1
        continue
print_tree(tree)













            
            
            




