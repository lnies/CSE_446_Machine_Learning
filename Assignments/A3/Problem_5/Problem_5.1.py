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
    Calculate the misclassification error (according to class canvas)
    """
    Yhat = np.dot(X,w)
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

def update_w(X, Y, w, eta, lamda, V):
    """
    Function for updating w to perform the stochastical descent with mini batches 
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Create mini batch consisting of 50 random data points
    M = 50
    mini_batch = np.array(0)
    mini_batch = np.delete(mini_batch, 0)
    Y_mini = np.array(0)
    Y_mini = np.delete(Y_mini, 0)
    for m in range(M):
        i = np.random.randint(0,N-1)
        mini_batch = np.append(mini_batch, X[i]).reshape(m+1,d)
        Y_mini = np.append(Y_mini, Y[i]).reshape(m+1,10)
    # Now transform features of mini-batch of M samples 
    mini_batch = np.sin(np.dot(2*mini_batch,V))
    # Add bias to data after mapping and mini batching
    mini_batch[:,49] = 1
    # Update w
    updated_w = w + eta * ( (1/M) * np.dot( np.transpose(mini_batch), ( Y_mini - np.dot(mini_batch, w) ) ) + np.dot(lamda, w) )
    return updated_w

def miniSGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev, V, k, lamda, eta, K):
    """
    Function for performing the mini batch stochastic gradient descent to find best w
    """
    # Define dimensionality and implement weight vector w
    d = len(Xtrain[0])
    w = np.zeros(k*10).reshape(k,10)
    # Initialize and create array for storing the norm 
    weight_norm = np.array(0)
    weight_norm = np.delete(weight_norm, 0)
    weight_norm = np.append(weight_norm, np.linalg.norm(w))
    # Generate fixed random index vector for calculating the errors
    error_indices_size_train = 3000 # Size of error array for train set
    error_indices_size_dev = 30 # Size of error array for dev set
    error_indices_size_test = 3000 # Size of error array for test set
    train_error_indices = np.random.choice(len(Xtrain),error_indices_size_train, replace=False)
    dev_error_indices = np.random.choice(len(Xdev),error_indices_size_dev, replace=False)
    test_error_indices = np.random.choice(len(Xtest),error_indices_size_test, replace=False)
    # Generate sample of X and Y to calculate error to save memory, already feature mapped
    Xtrain_error = Xtrain[train_error_indices]
    Xtrain_error = np.sin(np.dot(2*Xtrain_error,V))
    Xdev_error = Xdev[dev_error_indices]
    Xdev_error = np.sin(np.dot(2*Xdev_error,V))
    Xtest_error = Xtest[test_error_indices]
    Xtest_error = np.sin(np.dot(2*Xtest_error,V))
    Ytrain_error = Ytrain[train_error_indices]
    Ydev_error = Ydev[dev_error_indices]
    Ytest_error = Ytest[test_error_indices]
    # Initialize error arrays
    error_train = calc_avg_sq_err(Xtrain_error, Ytrain_error, w)
    error_dev = calc_avg_sq_err(Xdev_error, Ydev_error, w)
    error_test = calc_avg_sq_err(Xtest_error, Ytest_error, w)
    misc_train = calc_miscl_err(Ytrain_error, Xtrain_error, w)
    misc_dev = calc_miscl_err(Ydev_error, Xdev_error, w)
    misc_test = calc_miscl_err(Ytest_error, Xtest_error, w)
    print("Initial errors: Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_dev,misc_dev))
    # Start loop for finding optimal w
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
    for loop in range(K+1):
        # Update w for each loop
        w = update_w(Xtrain, Ytrain, w, eta, lamda, V)
        # Calculate errors and norm after every 500th iteration
        P = 500
        if ( loop % P == 0 and loop != 0):
            weight_norm = np.append(weight_norm, np.linalg.norm(w))
            error_train = calc_avg_sq_err(Xtrain_error, Ytrain_error, w)
            error_dev = calc_avg_sq_err(Xdev_error, Ydev_error, w)
            error_test = calc_avg_sq_err(Xtest_error, Ytest_error, w)
            misc_train = calc_miscl_err(Ytrain_error, Xtrain_error, w)
            misc_dev = calc_miscl_err(Ydev_error, Xdev_error, w)
            misc_test = calc_miscl_err(Ytest_error, Xtest_error, w)
            misc_train_arr = np.append(misc_train_arr, misc_train)
            misc_dev_arr = np.append(misc_dev_arr, misc_dev)
            misc_test_arr = np.append(misc_test_arr, misc_test)
            error_train_arr = np.append(error_train_arr, error_train)
            error_dev_arr = np.append(error_dev_arr, error_dev)
            error_test_arr = np.append(error_test_arr, error_test)
        # Print some runtime control
        if ( loop % 500 == 0 and loop != 0 ):
            print("Loop number %i/%i" % (loop,K))
            print("Squared Error: %3.5f, Miscl. Error %3.2f%%" % (error_test,misc_test))
    # Print out lowest errors
    print()
    print("+++ Results: +++")
    print("Lowest avg. sq. loss on training set: %3.5f at it. step %i" % (np.amin(error_train_arr), np.argmin(error_train_arr)*500))
    print("Lowest avg. sq. loss on dev set: %3.5f at it. step %i" % (np.amin(error_dev_arr), np.argmin(error_dev_arr)*500))
    print("Lowest avg. sq. loss on test set: %3.5f at it. step %i" % (np.amin(error_test_arr), np.argmin(error_test_arr)*500))
    print("Lowest miscl. error rate on training set: %3.2f%% at it. step %i" % (np.amin(misc_train_arr), np.argmin(misc_train_arr)*500))
    print("Lowest miscl. error rate on dev set: %3.2f%% at it. step %i" % (np.amin(misc_dev_arr), np.argmin(misc_dev_arr)*500))
    print("Lowest miscl. error rate on test set: %3.2f%% at it. step %i" % (np.amin(misc_test_arr), np.argmin(misc_test_arr)*500))
    print("Smallest number of total mistakes on training set: %i/60000 at it. step %i" % (np.amin(misc_train_arr)*60000/100, np.argmin(misc_train_arr)*500))
    print("Smallest number of total mistakes on dev set: %i/30 at it. step %i" % (np.amin(misc_dev_arr)*30/100, np.argmin(misc_dev_arr)*500))
    print("Smallest number of total mistakes on test set: %i/10000 at it. step %i" % (np.amin(misc_test_arr)*9970/100, np.argmin(misc_test_arr)*500))

    print()
    # Plot results
    x = np.arange(1,K/P+2,1)
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,3,1)
    #axes = plt.gca()
    #axes.set_ylim([0.01,0.1])
    plt.semilogy(x, error_train_arr, label = "Training error", linewidth=2)
    plt.semilogy(x, error_dev_arr, label = "Dev error", linewidth=2)
    plt.semilogy(x, error_test_arr, label = "Test error", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 10x axis value")
    plt.ylabel("Averaged squared error")
    plt.legend()       
    plt.subplot(1,3,2)
    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.plot(x, misc_train_arr, label = "Training misclassification", linewidth=2)
    plt.plot(x, misc_dev_arr, label = "Dev misclassification", linewidth=2)
    plt.plot(x, misc_test_arr, label = "Test misclassification", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Misclassification rate in %")
    plt.legend() 
    plt.subplot(1,3,3)
    plt.plot(x, weight_norm, label = "Norm of weight vector", linewidth=2)
    plt.grid(True)
    plt.xlabel("Iteration step K in 500x axis value")
    plt.ylabel("Norm of the weight vector")
    plt.legend() 
    plt.savefig("Problem_5.1_k_"+str(k)+"_"+str(lamda)+"_"+str(eta)+".png", bbox_inches='tight')
    plt.close(fig)

def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    with gzip.open("mnist_all_50pca_dims.gz") as f:
        data = pickle.load(f, encoding="bytes")
    Xtrain, Ytrain, Xtest, Ytest = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"]
    # Carve out small dev set from test set by randomly choose dev_size samples  
    dev_size = 30 # size of dev set
    dev_indices = np.random.choice(len(Xtest),dev_size, replace=False)
    Xdev = Xtest[dev_indices]
    Ydev = Ytest[dev_indices]
    ## Define parameters
    K = 30000 # Number of iterations
    k = 60000 # number of features
    V = np.random.randn(len(Xtrain[0]),k) # Create Gaussian distributed matrix V
    lamda = 1e-2 # Regularization parameter
    eta = 1e-3 # Stepsize
    ## Start stochastic gradient descent with mini batches
    miniSGD(Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev, V, k, lamda, eta, K)

if __name__ == '__main__':
    main()