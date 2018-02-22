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
    Calculate the misclassification error
    """
    # Define dimensionality
    d = len(Y)
    # Threshold 
    b = 0.5
    # Calculate the error
    """
    error = 0
    for i in range(d):
        if ( Y[i] != Y_est[i] ):
            error += 1
    error /= d
    error *= 100
    """
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
    error = ((1)/(2*N)) * np.square((np.linalg.norm( Y - np.dot(X,w) )))
    return error

def update_w(X, Y, w, eta, lamda):
    """
    Function for updating w to perform the descent
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Update w
    updated_w = w + np.dot(eta, ( np.dot( (1/N),np.dot( np.transpose(X), ( np.subtract(Y,np.dot(X, w))  ) ) + np.dot(lamda, w) )))
    return updated_w 

def GD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev):
    """
    Function for performing the gradient descent to find best w
    """
    # Define dimensionality
    d = len(Xtrain[0])
    # initialize starting stepsize eta(0) = 10 and starting w(0)=0
    w = np.zeros(d)
    eta = 1e-1 *(1/4) 
    lamda = 0.02
    error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
    misc_train = calc_miscl_err(Ytrain, Xtrain, w)
    print("Initial error: %3.5f" % error_train)
    # Start loop for finding optimal w
    K = 800 # Parameter to tune in order to find good convergence
    error_train_arr = np.array(0)
    error_train_arr = np.delete(error_train_arr, 0)
    error_train_arr = np.append(error_train_arr, error_train)
    misc_train_arr = np.array(0)
    misc_train_arr = np.delete(misc_train_arr, 0)
    misc_train_arr = np.append(misc_train_arr, misc_train)
    for k in range(K):
        w = update_w(Xtrain, Ytrain, w, eta, lamda)
        misc_train = calc_miscl_err(Ytrain, Xtrain, w)
        misc_train_arr = np.append(misc_train_arr, misc_train)
        error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
        error_train_arr = np.append(error_train_arr, error_train)
        if ( k % 10 == 0 ):
            print("Loop number %i/%i" % (k,K))
            print("Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_train,misc_train))
    # Print error growth
    x = np.arange(0,K+1,1)
    plt.subplot(1,2,1)
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=4)
    plt.grid(True)
    plt.xlabel("Iteration step K")
    plt.ylabel("Averaged squared error")
    plt.legend()
    
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=4)
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
    # Add Bias to data
    Xtrain[:,783] = 1
    Xdev[:,783] = 1
    Xtest[:,783] = 1
    GD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev)
    


if __name__ == '__main__':
    main()





