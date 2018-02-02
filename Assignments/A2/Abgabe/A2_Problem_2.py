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
import math

def plot_err(training_error, test_error, MaxIter, args):
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
    plt.savefig(str(args.train)+'full_log.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig(str(args.train)+'.full.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(0,30)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig(str(args.train)+'.early.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(MaxIter-30,MaxIter)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig(str(args.train)+'.late.png', bbox_inches='tight')
    
def plot_inner_loop(inner_train_err, inner_test_err, length, args, wo):
    x = np.arange(length)
    plt.figure()
    plt.plot(x, inner_train_err, color="blue", label="Training error")
    plt.plot(x, inner_test_err, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Index of observation in epoch")
    plt.ylabel("Error rate per feature in %")
    plt.grid(True)
    plt.savefig(str(args.train)+'.inner_loop_at_'+str(wo)+'.png', bbox_inches='tight')

def perceptron_test(d_train, d_test, d_dev, w, b):
    """
    Calculates the error on the trainings set and the error on the testset
    """
    error = np.zeros(3)
    for i in range(len(d_train)):
        if ( np.sign(np.dot(w,d_train[i][1:]) + b) != d_train[i][0] ):
            error[0] += 1
    for i in range(len(d_test)):
        if ( np.sign(np.dot(w,d_test[i][1:]) + b) != d_test[i][0] ):
            error[1] += 1
    for i in range(len(d_dev)):
        if ( np.sign(np.dot(w,d_dev[i][1:]) + b) != d_dev[i][0] ):
            error[2] += 1
    return error

def train_perceptron(d_train, d_test, MaxIter, args):
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
    for it in range(MaxIter):
        inner_train_err = np.array(0)
        inner_test_err = np.array(0)
        smallest_margin = 100000
        # Randomly shuffle the elements in the training set after each epoch
        np.random.shuffle(d_train)
        # Inner Loop to train perceptron on each feature per epoch
        for i in range(len(d_train)):
            # Use dot product to calculate activation function 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
            # Test for a certain epoch the error after each iteration in inner loop
            if ( it == MaxIter/10 or it == 9/10 * MaxIter ):
                val = perceptron_test(d_train, d_test, d_test, w, b)
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
                    plot_inner_loop( inner_train_err, inner_test_err , len(d_train)+1, args, wo)
            # Test for the smallest margin
            if ( smallest_margin > np.abs(np.dot(w/np.linalg.norm(w),d_train[i][1:])) ):
                smallest_margin = np.abs(np.dot(w/np.linalg.norm(w),d_train[i][1:]))
        # Test if algorithm converges       
        if ( np.array_equal(w,old_w) == True and is_print == False ):
            is_print = True
            print("Algorithm converges in epoch: ", it+1)
            print("Smallest margin is", smallest_margin)  
        old_w = list(w) # weight vector befor it gets updated
        # Test of the performance of the current perceptron status
        val = perceptron_test(d_train, d_test, d_test, w, b)
        # and storing the information in an array to be plotted later
        training_error = np.append(training_error, val[0])
        test_error = np.append(test_error, val[1])
    if ( is_print == False ):
        print("Algorithm does not converge after %i steps" % MaxIter)
    # Normalize errors to get error rate in percent
    training_error /= ( len(d_train) * 0.01 )
    test_error /= ( len(d_test) * 0.01 )
    # Plotting the error rates
    plot_err(training_error, test_error, MaxIter, args)
    return(w, b)
    
def tune_perceptron(d_train, d_test, MaxIter, args):
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
        # Inner Loop to train perceptron on each feature per epoch
        for i in range(len(train_set)):
            # Use dot product to calculate activation function 
            a = np.dot(w,train_set[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( train_set[i][0] * a <= 0 ):
                w += train_set[i][0] * train_set[i][1:]
                b += train_set[i][0]
        # Test of the performance of the current perceptron status
        val = perceptron_test(train_set, d_test, dev_set, w, b)
        # and storing the information in an array to be plotted later
        training_error = np.append(training_error, val[0])
        dev_error = np.append(dev_error, val[2])
    # Normalize errors to get error rate in percent
    training_error /= ( len(train_set) * 0.01 )
    dev_error /= ( len(dev_set) * 0.01 )
    argmin_train = np.argmin(training_error)
    argmin_dev = np.argmin(dev_error)
    return(argmin_dev)
    
def train_tuned_perceptron(d_train, d_test, MaxIter_tuned, args):
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
    for it in range(MaxIter_tuned):
        # Randomly shuffle the elements in the training set after each epoch
        np.random.shuffle(train_set)
        # Inner Loop to train perceptron on each feature per epoch
        for i in range(len(train_set)):
            # Use dot product to calculate activation function 
            a = np.dot(w,train_set[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( train_set[i][0] * a <= 0 ):
                w += train_set[i][0] * train_set[i][1:]
                b += train_set[i][0]
    # Test of the performance of the current perceptron status
    val = perceptron_test(train_set, d_test, dev_set, w, b)
    return(val)
    
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
    print("++++++++++ START ++++++++++" )
    print(" ")
    print("Train Perceptron without any optimizing with full training set:" )
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
    w, b = train_perceptron(d_train, d_test, MaxIter, args)
    
    # Tune the perceptron by searching for the MaxIter which yields lowest error
    # Calculate MaxIter several times and compute mean 
    print(" ")
    print("Now tune the perceptron by taking 80% training data to train and 20% development data to find the best development error: ")
    MaxIter_tuned = 0
    runs = 5
    for i in range(runs):
        MaxIter_tuned += tune_perceptron(d_train, d_test, MaxIter, args)
    MaxIter_tuned /= runs
    MaxIter_tuned = math.ceil(MaxIter_tuned)
    print("Tuned amount of interations for the main perceptron:", MaxIter_tuned)
    val = train_tuned_perceptron(d_train, d_test, MaxIter_tuned, args)
    print(" ")
    print("Now test perceptron with amount of calculated iterations:")
    print("Training error: %3.1f%%, Test error: %3.1f%%, Development error: %3.1f%%" % (val[0], val[1], val[2]))
    #print(w, b)
    return 0
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    