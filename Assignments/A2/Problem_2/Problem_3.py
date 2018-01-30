# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:27:27 2018

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

def plot_err(training_error, test_error, MaxIter):
    """
    Plots the error rate of the
    """
    x = range(1, MaxIter+2 )
    print(x)
    print(np.shape(x))
    plt.figure()
    plt.plot(np.log10(x), training_error, color="blue", label="Training error")
    plt.plot(np.log10(x), test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch (log scale)")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.figure()
    x = np.asarray(x)
    sp_training = UnivariateSpline(x,training_error,s=1*10e+0)
    sp_test = UnivariateSpline(x,test_error,s=1*10e+1)
    plt.plot(x, sp_training(x), color="blue", label="Spline interpolation of Training error")
    plt.plot(x, sp_test(x), color="red", label="Spline interpolation of Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.show()
    
def plot_inner_loop(inner_train_err, inner_test_err, length):
    x = np.arange(length)
    plt.figure()
    plt.plot(x, inner_train_err, color="blue", label="Training error")
    plt.plot(x, inner_test_err, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Index of observation in epoch")
    plt.ylabel("Error rate per feature in %")
    plt.grid(True)
    plt.show()

def perceptron_test(d_train, d_test, w, b, old_weigths, votes):
    """
    Calculates the error on the trainings set and the error on the testset
    """
    error = np.zeros(2)
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
    return error

def perceptron_train(d_train, d_test, MaxIter):
    """
    Algorithm for training the perceptron (according to CIML Algortithm 5, Chapter 4)
    """
    w = np.zeros(len(d_train[0])-1) # Initialize weight vector
    training_error = np.array(0)
    test_error = np.array(0)
    b = 0 # Initialize bias
    a = 0  # Initialize activiation variable
    # Start the training
    for it in range(MaxIter):
        print("Iteration step:", it)
        inner_train_err = np.array(0)
        inner_test_err = np.array(0)
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
            """
            # Test for a certain epoch the error after each iteration in inner loop
            if ( it == MaxIter/10 or it == 9/10 * MaxIter ):
                val = perceptron_test(d_train, d_test, w, b)
                inner_train_err = np.append(inner_train_err, val[0])
                inner_test_err = np.append(inner_test_err, val[1])
                # Only print at the end of the inner loop
                if ( i == len(d_train)-1):
                    plot_inner_loop( inner_train_err, inner_test_err , len(d_train)+1)
                    print("Weight vector:", w)
            """
        # Save last weight vector in the list
        old_weights = np.append(old_weights, w).reshape(len(votes), len(w))
        # Test of the performance of the current perceptron status
        val = perceptron_test(d_train, d_test, w, b, old_weights, votes)
        # and storing the information in an array to be plotted later
        training_error = np.append(training_error, val[0])
        test_error = np.append(test_error, val[1])
    # Normalize errors to get error rate in percent
    training_error /= ( len(d_train) * 0.01 )
    test_error /= ( len(d_test) * 0.01 )
    # Plotting the error rates
    plot_err(training_error, test_error, MaxIter)
    return(w, b)

def main():
    """
    Main function reads in the files from commandline.
    Then starts the routines to train the Perceptron
    """
    # Generating parser to read the filenames (and verbosity level)
    parser = argparse.ArgumentParser(description='Training and testing the Perceptron. Written by Lukas Nies for CSE446 Assignment 2')
    parser.add_argument("--train", "-tr", type=str, required=False, help="Training data file")
    parser.add_argument("--test", "-te", type=str, required=False, help="Test data file")
    parser.add_argument("--niter", "-i", type=int, required=False, help="Maximal Number of Iterations")
    parser.add_argument("--verbosity", "-v", type=int, required=False, help="Increase output verbosity (might not be used in this program)")
    args = parser.parse_args()

    # Uncommend this when using command line please
    """
    print( args.train, args.test, args.niter, args.verbosity )
    if ( Path(str(args.train)).is_file() != True ):
        print(" WARNING: Training file can not be found! ")
        return 0
    if ( Path(str(args.test)).is_file() != True ):
        print(" WARNING: Test file can not be found! ")
        return 0
    # Read data from ttraining data
    d_train = np.genfromtxt(args.train)
    d_test = np.genfromtxt(args.test)
    MaxIter = args.niter
    """
    d_train = np.genfromtxt("A2.9.train.tsv")
    d_test = np.genfromtxt("A2.9.test.tsv")
    MaxIter = 500 
    
    # Train the perceptron
    w, b = perceptron_train(d_train, d_test, MaxIter)
    #print(w, b)
    return 0
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    