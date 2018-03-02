# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:25:46 2018

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
import gzip
import pickle
import matplotlib.image as mpimg
from timeit import default_timer as timer

def calc_miscl_err(Y, X, w):
    """
    Calculate the misclassification error as posted on class canvas
    """
    # Define dimensionality
    d = len(Y)
    # Threshold 
    b = 0.5
    # Calculate the error
    Yhat = np.dot(X,w)
    Yhat_labels = ( Yhat - b ) >= 0
    errors = np.abs(Yhat_labels - Y)
    return (100*sum(errors)/(Yhat.shape[0]*1.0))
    
def calc_avg_sq_err(X, Y, w):
    """
    Function for calculating the average squared error
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Calculate the error
    error = ((1)/(2*N)) * (np.linalg.norm( Y - np.dot(X,w) ))**2
    return error

def calc_log_loss(X, Y, w):
    """
    Function to calculate the log loss 
    """
    # Define dimensionality and values
    N = len(X)
    d = len(X[0])
    Yhat = np.dot(X,w)
    ones = np.ones(N)
    # Calculate the loss
    loss = (1/N) * ( np.dot(Y,np.log(np.absolute(Yhat))) + np.dot(( ones - Y), np.log(np.absolute(ones-Yhat))))
    return loss

def gen_pol_feat(Xtrain_small, Xdev_small, Xtest_small):
    ## Function taken from class canvas (post by Edith Heiter)
    # Compute nxdxd matrices containing the outer product x^Tx for all n entries
    Xtrain_outer = np.einsum('ij, ik -> ijk', Xtrain_small, Xtrain_small, optimize=True)
    Xdev_outer = np.einsum('ij, ik -> ijk', Xdev_small, Xdev_small, optimize=True)
    Xtest_outer = np.einsum('ij, ik -> ijk', Xtest_small, Xtest_small, optimize=True)
    # Compute arrays of indices for upper triangle part of a dxd matrix
    j, k = np.triu_indices(Xtrain_small.shape[1])
    # Flatten the three dimensional array and keep only the upper triangle
    Xtrain = Xtrain_outer[:, j, k]
    Xdev = Xdev_outer[:, j, k]
    Xtest = Xtest_outer[:, j, k]
    
    return Xtrain, Xdev, Xtest

def update_w(X, Y, w, eta, lamda):
    """
    Function for updating w to perform the stochastical descent with mini batches 
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Create mini batch consisting of 100 random data points
    M = 100
    mini_batch = np.array(0)
    mini_batch = np.delete(mini_batch, 0)
    Y_mini = np.array(0)
    Y_mini = np.delete(Y_mini, 0)
    for m in range(M):
        i = np.random.randint(0,N-1)
        mini_batch = np.append(mini_batch, X[i]).reshape(m+1,d)
        Y_mini = np.append(Y_mini, Y[i])
    # Update w
    updated_w = w + eta * ( (1/M) * np.dot( np.transpose(mini_batch), ( Y_mini - np.dot(mini_batch, w) ) ) + np.dot(lamda, w) )
    return updated_w

def SGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev):
    """
    Function for performing the stochastic gradient descent to find best w
    """
    # Define dimensionality
    d = len(Xtrain[0])
    # initialize starting stepsize and parameterization
    w = np.zeros(d)
    eta = 1e-2 *(1/4)
    lamda = 0.01
    error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
    error_dev = calc_avg_sq_err(Xdev, Ydev, w)
    error_test = calc_avg_sq_err(Xtest, Ytest, w)
    misc_train = calc_miscl_err(Ytrain, Xtrain, w)
    misc_dev = calc_miscl_err(Ydev, Xdev, w)
    misc_test = calc_miscl_err(Ytest, Xtest, w)
    #log_loss_train = calc_log_loss(Xtrain, Ytrain, w)
    #log_loss_dev = calc_log_loss(Xdev, Ydev, w)
    #log_loss_test = calc_log_loss(Xtest, Ytest, w)
    print("Initial error: %3.5f" % error_train)
    # Start loop for finding optimal w
    K = 30000 # Parameter to tune in order to find good convergence
    error_train_arr = np.array(0)
    error_train_arr = np.delete(error_train_arr, 0)
    error_train_arr = np.append(error_train_arr, error_train)
    error_dev_arr = np.array(0)
    error_dev_arr = np.delete(error_dev_arr, 0)
    error_dev_arr = np.append(error_dev_arr, error_dev)
    error_test_arr = np.array(0)
    error_test_arr = np.delete(error_test_arr, 0)
    error_test_arr = np.append(error_test_arr, error_test)
    misc_train_arr = np.array(0)
    misc_train_arr = np.delete(misc_train_arr, 0)
    misc_train_arr = np.append(misc_train_arr, misc_train)
    misc_dev_arr = np.array(0)
    misc_dev_arr = np.delete(misc_dev_arr, 0)
    misc_dev_arr = np.append(misc_dev_arr, misc_dev)
    misc_test_arr = np.array(0)
    misc_test_arr = np.delete(misc_test_arr, 0)
    misc_test_arr = np.append(misc_test_arr, misc_test)
    log_train_arr = np.array(0)
    log_train_arr = np.delete(log_train_arr, 0)
    log_train_arr = np.append(log_train_arr, 0)
    log_test_arr = np.array(0)
    log_test_arr = np.delete(log_test_arr, 0)
    log_test_arr = np.append(log_test_arr, 0)
    log_dev_arr = np.array(0)
    log_dev_arr = np.delete(log_dev_arr, 0)
    log_dev_arr = np.append(log_dev_arr, 0)

    for k in range(K+1):
        # Update w for each loop
        w = update_w(Xtrain, Ytrain, w, eta, lamda)
        # Print some runtime control
        if ( k % 1000 == 0 and k != 0 ):
            print("Loop number %i/%i" % (k,K))
            print("Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_dev,misc_dev))
        # Calculate errors after every 500th iteration
        P = 500
        if ( k % P == 0 and k != 0):
            log_loss_train = calc_log_loss(Xtrain, Ytrain, w)
            log_loss_dev = calc_log_loss(Xdev, Ydev, w)
            log_loss_test = calc_log_loss(Xtest, Ytest, w)
            log_train_arr = np.append(log_train_arr, log_loss_train)
            log_test_arr = np.append(log_test_arr, log_loss_test)
            log_dev_arr = np.append(log_dev_arr, log_loss_dev)
            misc_train = calc_miscl_err(Ytrain, Xtrain, w)
            misc_dev = calc_miscl_err(Ydev, Xdev, w)
            misc_test = calc_miscl_err(Ytest, Xtest, w)
            misc_train_arr = np.append(misc_train_arr, misc_train)
            misc_dev_arr = np.append(misc_dev_arr, misc_dev)
            misc_test_arr = np.append(misc_test_arr, misc_test)
            error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
            error_train_arr = np.append(error_train_arr, error_train)
            error_dev = calc_avg_sq_err(Xdev, Ydev, w)
            error_dev_arr = np.append(error_dev_arr, error_dev)
            error_test = calc_avg_sq_err(Xtest, Ytest, w)
            error_test_arr = np.append(error_test_arr, error_test)
    # Plot the results 
    x = np.arange(1,K/P+2,1)
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,3,1)
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=2)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=2)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Averaged squared error")
    plt.legend()
    plt.subplot(1,3,2)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=2)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=2)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Misclassification rate in %")
    plt.legend() 
    plt.subplot(1,3,3)
    plt.plot(x, log_train_arr, label = "Training log loss", linewidth=2)
    plt.plot(x, log_dev_arr, label = "Dev log loss", linewidth=2)
    plt.plot(x, log_test_arr, label = "Test log loss", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Log Loss")
    plt.legend() 
    plt.savefig("Problem_2.1_pol_features.png", bbox_inches='tight')
    plt.close(fig)

def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    with gzip.open("mnist_2_vs_9_with_40_PCA_features.gz") as f:
        data = pickle.load(f, encoding="bytes")
    Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]
    # Generate new feature vectors
    Xtrain_pol, Xdev_pol, Xtest_pol  = gen_pol_feat(Xtrain, Xdev, Xtest)
    # Add Bias to data
    Xtrain_pol[:,819] = 1
    Xdev_pol[:,819] = 1
    Xtest_pol[:,819] = 1
    # Use gradient descent for searching for optimum
    SGD(Xtrain_pol, Ytrain, Xtest_pol, Ytest, Xdev_pol, Ydev)

if __name__ == '__main__':
    main()