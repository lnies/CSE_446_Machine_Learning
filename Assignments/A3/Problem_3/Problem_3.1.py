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

def calc_miscl_err(Y, Yhat):
    """
    Calculate the misclassification error (according to class canvas)
    """
    indsYhat=np.argmax(Yhat,axis=1)
    indsY=np.argmax(Y,axis=1)
    errors = (indsYhat-indsY)!=0
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
    Function for updating w to perform the descent
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Update w
    updated_w = w + eta * ( (1/N) * np.dot( np.transpose(X), ( Y - np.dot(X, w) ) ) + np.dot(lamda, w) )
    return updated_w
    
def SGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev):
    """
    Function for performing the stochastic gradient descent to find best w
    """
    # Define dimensionality
    d = len(Xtrain[0])
    # initialize starting stepsize eta(0) = 10 and starting w(0)=0
    w = np.zeros(d*10).reshape(d,10)
    eta = 1e-1 * (2/4) 
    lamda = 0.01
    error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
    error_dev = calc_avg_sq_err(Xdev, Ydev, w)
    error_test = calc_avg_sq_err(Xtest, Ytest, w)
    Yhat_train = np.dot(Xtrain,w)
    Yhat_dev = np.dot(Xdev,w)
    Yhat_test = np.dot(Xtest,w)
    misc_train = calc_miscl_err(Ytrain, Yhat_train)
    misc_dev = calc_miscl_err(Ydev, Yhat_dev)
    misc_test = calc_miscl_err(Ytest, Yhat_test)
    print("Initial error: %3.5f" % error_train)
    # Start loop for finding optimal w
    K = 800 # Parameter to tune in order to find good convergence
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
        if ( k % 10 == 0 and k != 0 ):
            print("Loop number %i/%i" % (k,K))
            print("Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_dev,misc_dev))
        # Calculate errors after every Pth iteration
        P = 10
        if ( k % P == 0 and k != 0):
            Yhat_train = np.dot(Xtrain,w)
            Yhat_dev = np.dot(Xdev,w)
            Yhat_test = np.dot(Xtest,w)
            misc_train = calc_miscl_err(Ytrain, Yhat_train)
            misc_dev = calc_miscl_err(Ydev, Yhat_dev)
            misc_test = calc_miscl_err(Ytest, Yhat_test)
            misc_train_arr = np.append(misc_train_arr, misc_train)
            misc_dev_arr = np.append(misc_dev_arr, misc_dev)
            misc_test_arr = np.append(misc_test_arr, misc_test)
            error_train = calc_avg_sq_err(Xtrain, Ytrain, w)
            error_train_arr = np.append(error_train_arr, error_train)
            error_dev = calc_avg_sq_err(Xdev, Ydev, w)
            error_dev_arr = np.append(error_dev_arr, error_dev)
            error_test = calc_avg_sq_err(Xtest, Ytest, w)
            error_test_arr = np.append(error_test_arr, error_test)
            
    # Plot the graphs
    x = np.arange(1,K/P+2,1)
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=2)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=2)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 10x axis value")
    plt.ylabel("Averaged squared error")
    plt.legend()       
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([10,20])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=2)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=2)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 10x axis value")
    plt.ylabel("Misclassification rate in %")
    plt.legend()
    # Save plot in extra file 
    plt.savefig("Problem_3.1.png", bbox_inches='tight')
    plt.close(fig)

def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    # Open and save the dataset from NIST
    with gzip.open("mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    # Configure Sets
    Xtrain = train_set[0]
    Xdev = valid_set[0]
    Xtest = test_set[0]
    train_label = train_set[1]
    dev_label = valid_set[1]
    test_label = test_set[1]
    # Add Bias to data
    Xtrain[:,783] = 1
    Xdev[:,783] = 1
    Xtest[:,783] = 1
    # Create label matrix
    Ytrain = np.zeros(len(Xtrain)*10).reshape(len(Xtrain),10)
    Ytest = np.zeros(len(Xtest)*10).reshape(len(Xtest),10)
    Ydev = np.zeros(len(Xdev)*10).reshape(len(Xdev),10)
    # Fill label Matrix
    for i in range(len(Xtrain)):
        Ytrain[i,train_label[i]] = 1
    for i in range(len(Xtest)):
        Ytest[i,test_label[i]] = 1
    for i in range(len(Xdev)):
        Ydev[i,dev_label[i]] = 1
    # Start stochastic gradient descent  
    SGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev)
    
if __name__ == '__main__':
    main()