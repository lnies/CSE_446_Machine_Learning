# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:45:11 2018

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
    plt.ylabel("Errors rate per Epoch in %")
    plt.grid(True)
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Errors rate per Epoch in %")
    plt.grid(True)
    plt.figure()
    x = np.asarray(x)
    sp_training = UnivariateSpline(x,training_error,s=1*10e+0)
    sp_test = UnivariateSpline(x,test_error,s=1*10e+1)
    plt.plot(x, sp_training(x), color="blue", label="Spline interpolation of Training error")
    plt.plot(x, sp_test(x), color="red", label="Spline interpolation of Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Errors rate per Epoch in %")
    plt.grid(True)
    plt.show()

def perceptron_test(d_train, d_test, w, b):
    """
    Calculates the error on the trainings set and the error on the testset
    """
    error = np.zeros(2)
    for i in range(len(d_train)):
        if ( np.sign(np.dot(w,d_train[i][1:]) + b) != d_train[i][0] ):
            error[0] += 1
    for i in range(len(d_test)):
        if ( np.sign(np.dot(w,d_test[i][1:]) + b) != d_test[i][0] ):
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
        for i in range(len(d_train)): 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
        # Test of the performance of the current perceptron status
        val = perceptron_test(d_train, d_test, w, b)
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
    parser.add_argument("--train", "-tr", type=str, required=True, help="Training data file")
    parser.add_argument("--test", "-te", type=str, required=True, help="Test data file")
    parser.add_argument("--niter", "-i", type=int, required=True, help="Maximal Number of Iterations")
    parser.add_argument("--verbosity", "-v", type=int, required=False, help="Increase output verbosity (might not be used in this program)")
    args = parser.parse_args()
    """
    Uncommend this when using command line please
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
    d_train = np.genfromtxt("A2.2.train.tsv")
    d_test = np.genfromtxt("A2.2.test.tsv")
    """
    # Train the perceptron
    w, b = perceptron_train(d_train, d_test, MaxIter)
    #print(w, b)
    return 0
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    