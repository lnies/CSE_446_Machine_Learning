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

def calc_miscl_err(Y, Y_est):
    """
    Calculate the misclassification error
    """
    # Define dimensionality
    d = len(Y)
    # Calculate the error
    error = 0
    for i in range(d):
        if ( Y[i] != Y_est[i] ):
            error += 1
    error /= d
    error *= 100
    # Return error
    return error
    
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

def estimate(X, Y, wstar, TH):
    """
    Function for estimating the true label
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Estimate the label 
    Y_est = np.zeros(N)
    for i in range(N):
        if ( np.dot(X[i],wstar) >= TH ) :
            Y_est[i] = 1
        else:
            Y_est[i] = 0
    # Return estimation
    return Y_est

def update_w(X, Y, w, eta, lamda):
    """
    Function for updating w to perform the descent
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Update w
    updated_w = w + eta * ( (1/N) * np.dot( np.transpose(X), ( Y - np.dot(X, w) ) ) + np.dot(lamda, w) )
    return updated_w

def GD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev):
    """
    Function for performing the gradient descent to find best w
    """
    # Define dimensionality
    d = len(Xtrain[0])
    # initialize starting stepsize eta(0) = 10 and starting w(0)=0
    w = np.zeros(d)
    eta = 1e-2 
    lamda = 1
    error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
    error_dev = calc_avg_sq_err(Xdev, Ydev, w)
    error_test = calc_avg_sq_err(Xtest, Ytest, w)
    Y_est_train = estimate(Xtrain, Ytrain, w, TH = 0.5)
    Y_est_dev = estimate(Xdev, Ydev, w, TH = 0.5)
    Y_est_test = estimate(Xtest, Ytest, w, TH = 0.5)
    misc_train = calc_miscl_err(Ytrain, Y_est_train)
    misc_dev = calc_miscl_err(Ydev, Y_est_dev)
    misc_test = calc_miscl_err(Ytest, Y_est_test)
    print("Initial error: %3.5f" % error_train)
    # Start loop for finding optimal w
    K = 400 # Parameter to tune in order to find good convergence
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
    for k in range(K):
        w = update_w(Xtrain, Ytrain, w, eta, lamda)
        Y_est_train = estimate(Xtrain, Ytrain, w, TH = 0.5)
        Y_est_dev = estimate(Xdev, Ydev, w, TH = 0.5)
        Y_est_test = estimate(Xtest, Ytest, w, TH = 0.5)
        misc_train = calc_miscl_err(Ytrain, Y_est_train)
        misc_dev = calc_miscl_err(Ydev, Y_est_dev)
        misc_test = calc_miscl_err(Ytest, Y_est_test)
        misc_train_arr = np.append(misc_train_arr, misc_train)
        misc_dev_arr = np.append(misc_dev_arr, misc_dev)
        misc_test_arr = np.append(misc_test_arr, misc_test)
        error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
        error_train_arr = np.append(error_train_arr, error_train)
        error_dev = calc_avg_sq_err(Xdev, Ydev, w)
        error_dev_arr = np.append(error_dev_arr, error_dev)
        error_test = calc_avg_sq_err(Xtest, Ytest, w)
        error_test_arr = np.append(error_test_arr, error_test)
        if ( k % 10 == 0 ):
            print("Loop number %i/%i" % (k,K))
            print("Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_dev,misc_dev))
    # Print error growth
    x = np.arange(0,K+1,1)
    plt.subplot(1,2,1)
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=4)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=4)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=4)
    plt.grid(True)
    plt.xlabel("Iteration step K")
    plt.ylabel("Averaged squared error")
    plt.legend()
    
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=4)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=4)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=4)
    plt.grid(True)
    plt.xlabel("Iteration step K")
    plt.ylabel("Misclassification rate in %")
    plt.legend()
    plt.show()





def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    with gzip.open("mnist_2_vs_9.gz") as f:
        data = pickle.load(f, encoding="bytes")
    Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]
    
    GD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev)
    
    
    




if __name__ == '__main__':
    main()





