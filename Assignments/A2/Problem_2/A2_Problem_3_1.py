# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:31:21 2018

@author: Lukas
"""

"""
Import of external tools
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import math

def plot_err(training_error, test_error, MaxIter):
    """
    Plots the error rate of the
    """
    x = range(1, MaxIter+1 )
    plt.figure()
    plt.plot(np.log10(x), training_error, color="blue", label="Training error")
    plt.plot(np.log10(x), test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch (log scale)")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'full_log.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.full.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(0,30)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.early.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(MaxIter-30,MaxIter)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.late.png', bbox_inches='tight')
    
def plot_inner_loop(inner_train_err, inner_test_err, length, wo):
    x = np.arange(length)
    plt.figure()
    plt.plot(x, inner_train_err, color="blue", label="Training error")
    plt.plot(x, inner_test_err, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Index of observation in epoch")
    plt.ylabel("Error rate per feature in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.inner_loop_at_'+str(wo)+'.png', bbox_inches='tight')

def perceptron_test(d_train, d_test, d_dev, w, b, old_weigths, votes):
    """
    Calculates the error on the trainings set and the error on the testset
    weigthed with the votes of the total of all weight vectors
    """
    error = np.zeros(3)
    for i in range(len(d_train)):
        activation = 0
        for j in range(len(old_weigths)):
            activation += votes[j] * np.sign(np.dot(old_weigths[j],d_train[i][1:]) + b) 
        if ( np.sign(activation) != d_train[i][0] ):
            error[0] += 1
    for i in range(len(d_test)):
        activation = 0
        for j in range(len(old_weigths)):
            activation += votes[j] * np.sign(np.dot(old_weigths[j],d_test[i][1:]) + b) 
        if ( np.sign(activation) != d_test[i][0] ):
            error[1] += 1
    for i in range(len(d_dev)):
        activation = 0
        for j in range(len(old_weigths)):
            activation += votes[j] * np.sign(np.dot(old_weigths[j],d_dev[i][1:]) + b) 
        if ( np.sign(activation) != d_dev[i][0] ):
            error[2] += 1
    return error

def train_perceptron(d_train, d_test, MaxIter):
    """
    Algorithm for training the perceptron (according to CIML Algortithm 5, Chapter 4)
    """
    w = np.zeros(len(d_train[0])-1) # Initialize weight vector
    old_w = np.zeros(len(d_train[0]-1))
    d = len(d_train[0])-1
    n = len(d_train)
    training_error = np.array(0)
    training_error = np.delete(training_error, 0)
    test_error = np.array(0)
    test_error = np.delete(test_error, 0)
    b = 0 # Initialize bias
    a = 0  # Initialize activiation variable
    is_print = False
    # Create array to store the information for the margin
    # Start the training
    old_w = list(w) # weight vector befor it gets updated
    old_weights = np.array(0)
    old_weights = np.delete(old_weights, 0)
    for it in range(MaxIter):
        print("Iteration step:", it)
        inner_train_err = np.array(0)
        inner_test_err = np.array(0)
        smallest_margin = 100000
        # Randomly shuffle the elements in the training set after each epoch
        np.random.shuffle(d_train)
        # Inner Loop to train perceptron on each feature per epoch
        # Implement voted perceptron
        votes = np.array(0).reshape(1,)
        old_weights = np.array(0)
        old_weights = np.delete(old_weights, 0)
        # If a weight vector gets updated immediatly then he gets one vote
        votes[0] = 1 
        for i in range(len(d_train)):
            #print(i)
            # Use dot product to calculate activation function 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                # Update the votes vector for the new weight vector
                votes = np.append(votes, 1)
                # Store old weightvector on array of old weight vectors
                old_weights = np.append(old_weights, w).reshape(len(votes)-1, len(w))
                # Update old weight vector to get a new one
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
            else:
                # If prediction is correct, don't update weight vector and increase current counts for vectro by 1
                votes[len(votes)-1] += 1
            # Test for a certain epoch the error after each iteration in inner loop
            if ( it == MaxIter/10 or it == 9/10 * MaxIter ):
                val = perceptron_test(d_train, d_test, d_test, w, b, old_weights, votes)
                inner_train_err = np.append(inner_train_err, val[0])
                inner_test_err = np.append(inner_test_err, val[1])
                # Only print at the end of the inner loop
                if ( i == len(d_train)-1):
                    if ( it == MaxIter/10 ):
                        wo = 10
                        w_norm = w/np.linalg.norm(w)
                        print("Normed weight vector at 10 percent:", w_norm)
                    if ( it == 9/10 * MaxIter ):
                        wo = 90
                        w_norm = w/np.linalg.norm(w)
                        print("Normed weight vector at 90 percent:", w_norm)
                    plot_inner_loop( inner_train_err, inner_test_err , len(d_train)+1, wo)
            # Test for the smalles margin
            if ( smallest_margin > np.abs(np.dot(w,d_train[i][1:])) ):
                smallest_margin = np.abs(np.dot(w,d_train[i][1:]))
        # Test if algorithm converges       
        if ( np.array_equal(w,old_w) == True and is_print == False ):
            is_print = True
            print("Algorithm converges in epoch: ", it+1)
            print("Smallest margin is", smallest_margin)  
        old_w = list(w) # weight vector befor it gets updated
        # Save last weight vector in the list
        old_weights = np.append(old_weights, w).reshape(len(votes), len(w))
        # Test of the performance of the current perceptron status
        val = perceptron_test(d_train, d_test, d_test, w, b, old_weights, votes)
        # and storing the information in an array to be plotted later
        training_error = np.append(training_error, val[0])
        test_error = np.append(test_error, val[1])
    if ( is_print == False ):
        print("Algorithm does not converge after %i steps" % MaxIter)
    # Normalize errors to get error rate in percent
    training_error /= ( len(d_train) * 0.01 )
    test_error /= ( len(d_test) * 0.01 )
    # Plotting the error rates
    plot_err(training_error, test_error, MaxIter)
    return(w, b)
    
def tune_perceptron(d_train, d_test, MaxIter):
    """
    Algorithm tuning the maximum number of iterations based on the algorithm 
    for training the perceptron (according to CIML Algortithm 5, Chapter 4)
    """
    w = np.zeros(len(d_train[0])-1) # Initialize weight vector
    old_w = np.zeros(len(d_train[0]-1))
    d = len(d_train[0])-1
    n = len(d_train)
    training_error = np.array(0)
    training_error = np.delete(training_error, 0)
    dev_error = np.array(0)
    dev_error = np.delete(dev_error, 0)
    # Slice trainings set into training and development set: 80% training, 20% development
    train_set = d_train[:][0:(int(8*n/10))]
    dev_set = d_train[:][int(8*n/10):n]
    b = 0 # Initialize bias
    a = 0  # Initialize activiation variable
    # Start the training
    for it in range(MaxIter):
        # Randomly shuffle the elements in the training set after each epoch
        np.random.shuffle(train_set)
        # Implement voted perceptron
        votes = np.array(0).reshape(1,)
        old_weights = np.array(0)
        old_weights = np.delete(old_weights, 0)
        # If a weight vector gets updated immediatly then he gets one vote
        votes[0] = 1 
        # Inner Loop to train perceptron on each feature per epoch
        for i in range(len(d_train)):
            # Use dot product to calculate activation function 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                # Update the votes vector for the new weight vector
                votes = np.append(votes, 1)
                # Store old weightvector on array of old weight vectors
                old_weights = np.append(old_weights, w).reshape(len(votes)-1, len(w))
                # Update old weight vector to get a new one
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
            else:
                # If prediction is correct, don't update weight vector and increase current counts for vectro by 1
                votes[len(votes)-1] += 1
        # Test of the performance of the current perceptron status
        val = perceptron_test(train_set, d_test, dev_set, w, b, old_weights, votes)
        # and storing the information in an array to be plotted later
        training_error = np.append(training_error, val[0])
        dev_error = np.append(dev_error, val[2])
    # Normalize errors to get error rate in percent
    training_error /= ( len(train_set) * 0.01 )
    dev_error /= ( len(dev_set) * 0.01 )
    argmin_train = np.argmin(training_error)
    argmin_dev = np.argmin(dev_error)
    return(argmin_dev)
    
def train_tuned_perceptron(d_train, d_test, MaxIter_tuned):
    """
    Algorithm for training the tuned perceptron (according to CIML Algortithm 5, Chapter 4)
    """
    w = np.zeros(len(d_train[0])-1) # Initialize weight vector
    old_w = np.zeros(len(d_train[0]-1))
    d = len(d_train[0])-1
    n = len(d_train)
    dev_error = np.array(0)
    dev_error = np.delete(dev_error, 0)
    # Slice trainings set into training and development set: 80% training, 20% development
    train_set = d_train[:][0:(int(8*n/10))]
    dev_set = d_train[:][int(8*n/10):n]
    b = 0 # Initialize bias
    a = 0  # Initialize activiation variable
    # Start the training
    old_weights = np.array(0)
    old_weights = np.delete(old_weights, 0)
    votes = np.array(0).reshape(1,)
    for it in range(MaxIter_tuned):
        # Randomly shuffle the elements in the training set after each epoch
        np.random.shuffle(train_set)
        # Implement voted perceptron
        votes = np.array(0).reshape(1,)
        old_weights = np.array(0)
        old_weights = np.delete(old_weights, 0)
        # If a weight vector gets updated immediatly then he gets one vote
        votes[0] = 1 
        # Inner Loop to train perceptron on each feature per epoch
        for i in range(len(d_train)):
            # Use dot product to calculate activation function 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                # Update the votes vector for the new weight vector
                votes = np.append(votes, 1)
                # Store old weightvector on array of old weight vectors
                old_weights = np.append(old_weights, w).reshape(len(votes)-1, len(w))
                # Update old weight vector to get a new one
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
            else:
                # If prediction is correct, don't update weight vector and increase current counts for vectro by 1
                votes[len(votes)-1] += 1
    # Test of the performance of the current perceptron status
    val = perceptron_test(train_set, d_test, dev_set, w, b, old_weights, votes)
    return(val)
    
def main():
    """
    Main function reads in the files.
    Then starts the routines to train the Perceptron
    """
    
    """
    Uncommend this when using command line please
    """
    print("++++++++++ START ++++++++++" )
    print(" ")
    print("Train Perceptron without any optimizing with full training set:" )
    
    # Read data from ttraining data
    #d_train_1 = np.genfromtxt("A2.6.train.tsv")
    #d_train_2 = np.genfromtxt("A2.7.train.tsv")
    #d_train_3 = np.genfromtxt("A2.8.train.tsv")
    # Combine the three data files to greate a larger trainings set
    #d_train_1 = np.append(d_train_1, d_train_2).reshape(len(d_train_1)+len(d_train_2), len(d_train_1[0]))
    #d_train_1 = np.append(d_train_1, d_train_3).reshape(len(d_train_1)+len(d_train_3), len(d_train_1[0]))
    #d_train = d_train_1[:]
    
    d_train = np.genfromtxt("A2.5.train.tsv")
    d_test = np.genfromtxt("A2.5.test.tsv")
    MaxIter = 549

    # Train the perceptron
    w, b = train_perceptron(d_train, d_test, MaxIter)
    
    # Tune the perceptron by searching for the MaxIter which yields lowest error
    # Calculate MaxIter several times and compute mean 
    print(" ")
    print("Now tune the perceptron by taking 80% training data to train and 20% development data to find the best development error: ")
    MaxIter_tuned = 0
    runs = 1
    for i in range(runs):
        MaxIter_tuned += tune_perceptron(d_train, d_test, MaxIter)
    MaxIter_tuned /= runs
    MaxIter_tuned = math.ceil(MaxIter_tuned)
    print("Tuned amount of interations for the main perceptron:", MaxIter_tuned)
    val = train_tuned_perceptron(d_train, d_test, MaxIter_tuned)
    print(" ")
    print("Now test perceptron with amount of calculated iterations:")
    print("Training error: %3.1f%%, Test error: %3.1f%%, Development error: %3.1f%%" % (val[0], val[1], val[2]))
    #print(w, b)

    return 0
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    