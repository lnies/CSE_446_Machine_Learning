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

def miniSGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev):
    """
    Function for performing the mini batch stochastic gradient descent to find best w
    """
    # Define dimensionality
    d = len(Xtrain[0])
    # initialize starting stepsize eta(0) = 10 and starting w(0)=0
    w = np.zeros(d)
    eta = 1e-2 *(1/4) 
    lamda = 0.005
    error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
    error_dev = calc_avg_sq_err(Xdev, Ydev, w)
    error_test = calc_avg_sq_err(Xtest, Ytest, w)
    misc_train = calc_miscl_err(Ytrain, Xtrain, w)
    misc_dev = calc_miscl_err(Ydev, Xdev, w)
    misc_test = calc_miscl_err(Ytest, Xtest, w)
    print("Initial error: %3.5f" % error_train)
    # Start loop for finding optimal w
    K = 100000 # Parameter to tune in order to find good convergence
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
            
            
    x = np.arange(1,K/P+2,1)
    fig = plt.figure(figsize=(16,6))
            
    plt.subplot(1,2,1)
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=2)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=2)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Averaged squared error")
    plt.legend()
            
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=2)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=2)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Misclassification rate in %")
    plt.legend()
        
    plt.savefig("Problem_1.4.png", bbox_inches='tight')

    plt.close(fig)

def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    with gzip.open("mnist_2_vs_9.gz") as f:
        data = pickle.load(f, encoding="bytes")
    Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]
    
    miniSGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev)
    
    
    




if __name__ == '__main__':
    main()





