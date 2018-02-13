# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:59:50 2018

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


def plot_10(X):
    """
    Plot some digits from X
    """
    
    f, axarr = plt.subplots(1, 6)
    axarr[0].imshow(X[1].reshape(28,28), cmap='gray')
    axarr[0].set_title('Digit 0')
    axarr[1].imshow(np.transpose(X[1].reshape(28,28)), cmap='gray')
    axarr[1].set_title('Digit 1')
    axarr[2].imshow(np.dot(np.transpose(X[1].reshape(28,28)),X[1].reshape(28,28)), cmap='gray')
    axarr[2].set_title('Digit 0')
    axarr[3].imshow(X[19].reshape(28,28), cmap='gray')
    axarr[3].set_title('Digit 0')
    axarr[4].imshow(np.transpose(X[19].reshape(28,28)), cmap='gray')
    axarr[4].set_title('Digit 1')
    axarr[5].imshow(np.dot(np.transpose(X[19].reshape(28,28)),X[19].reshape(28,28)), cmap='gray')
    axarr[5].set_title('Digit 0')
    
    
    """
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(X[1].reshape(28,28), cmap='gray')
    axarr[0, 0].set_title('Digit 0')
    axarr[0, 1].imshow(X[3].reshape(28,28), cmap='gray')
    axarr[0, 1].set_title('Digit 1')
    axarr[0, 2].imshow(X[5].reshape(28,28), cmap='gray')
    axarr[0, 2].set_title('Digit 2')
    axarr[0, 3].imshow(X[7].reshape(28,28), cmap='gray')
    axarr[0, 3].set_title('Digit 3')
    axarr[0, 4].imshow(X[9].reshape(28,28), cmap='gray')
    axarr[0, 4].set_title('Digit 4')
    axarr[1, 0].imshow(X[11].reshape(28,28), cmap='gray')
    axarr[1, 0].set_title('Digit 5')
    axarr[1, 1].imshow(X[13].reshape(28,28), cmap='gray')
    axarr[1, 1].set_title('Digit 6')
    axarr[1, 2].imshow(X[15].reshape(28,28), cmap='gray')
    axarr[1, 2].set_title('Digit 7')
    axarr[1, 3].imshow(X[17].reshape(28,28), cmap='gray')
    axarr[1, 3].set_title('Digit 8')
    axarr[1, 4].imshow(X[19].reshape(28,28), cmap='gray')
    axarr[1, 4].set_title('Digit 9')
    """
    
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


def find_w(X, Y, lamda):
    """
    Function for finding wstar
    """
    # Define dimensionality
    N = len(X)
    d = len(X[0])
    # Create d dimensional idendity array
    ones = np.diag(np.ones(d))
    # Define lamda and decision threshold
    # Calculate wstar
    wstar = np.dot( np.linalg.inv( np.dot((1/N),np.dot(np.transpose(X),X)) + np.dot((1/N),np.dot(lamda,ones))), np.dot( (1/N), np.dot( np.transpose(X),Y) ) ) 
    # Return labels and vector wstar
    return wstar


def main():
    """
    Main function: main sequence of programm is controlled and called here
    """
    # Read in MNIST data    
    with gzip.open("mnist_2_vs_9.gz") as f:
        data = pickle.load(f, encoding="bytes")
    Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]
    # Loop over different values of lambda and threshold to tune 
    # the algorithm on the development set
    print("Begin of searching for best parameters of lambda and the threshold:")
    print("")
    lowest_sq_err_dev = 100000
    lowest_miscl_err_dev = 100000
    T_best = 0.5
    L_best = 0.5
    for T in range(1,11):
        print("Loop %i out of 10" % T)
        T /= 10
        for L in range(10,250):
            L /= 1
            # Calculate optimized vector w and estimation on Xtrain
            w_star_train = find_w(Xtrain, Ytrain, lamda = L)
            # Calculate the squared errors
            sq_err_train = calc_avg_sq_err(Xtrain, Ytrain, w_star_train)
            sq_err_dev = calc_avg_sq_err(Xdev, Ydev, w_star_train)
            sq_err_test = calc_avg_sq_err(Xtest, Ytest, w_star_train)
            # Esimate label and calculate misclassification error
            Ytrain_est = estimate(Xtrain, Ytrain, w_star_train, TH = T)
            Ydev_est = estimate(Xdev, Ydev, w_star_train, TH = T)
            Ytest_est = estimate(Xtest, Ytest, w_star_train, TH = T)
            miscl_err_train = calc_miscl_err(Ytrain, Ytrain_est)
            miscl_err_dev = calc_miscl_err(Ydev, Ydev_est)
            miscl_err_test = calc_miscl_err(Ytest, Ytest_est)
            # Rough test if error rates have improved
            if ( miscl_err_dev < lowest_miscl_err_dev or sq_err_dev < lowest_sq_err_dev  ):
                lowest_miscl_err_dev = miscl_err_dev
                lowest_sq_err_dev = sq_err_dev
                T_best = T
                L_best = L
    print("")
    print("After grid search the best parameters were found:")
    print("Threshold: %3.1f, Lambda: %3.1f" % (T_best, L_best))
    print("")
    # Recalculate best result
    # Calculate optimized vector w and estimation on Xtrain
    w_star_train = find_w(Xtrain, Ytrain, lamda = L_best)
    # Calculate the squared errors
    sq_err_train = calc_avg_sq_err(Xtrain, Ytrain, w_star_train)
    sq_err_dev = calc_avg_sq_err(Xdev, Ydev, w_star_train)
    sq_err_test = calc_avg_sq_err(Xtest, Ytest, w_star_train)
    # Esimate label and calculate misclassification error
    Ytrain_est = estimate(Xtrain, Ytrain, w_star_train, TH = T_best)
    Ydev_est = estimate(Xdev, Ydev, w_star_train, TH = T_best)
    Ytest_est = estimate(Xtest, Ytest, w_star_train, TH = T_best)
    miscl_err_train = calc_miscl_err(Ytrain, Ytrain_est)
    miscl_err_dev = calc_miscl_err(Ydev, Ydev_est)
    miscl_err_test = calc_miscl_err(Ytest, Ytest_est)
    print("Best parameters give:")
    print("Avgerage squared errors:")
    print("Train: %3.5f, Dev: %3.5f, Test: %3.5f" % (sq_err_train, sq_err_dev, sq_err_test))
    print("")
    print("Misclassification errors:")
    print("Train: %3.2f%%, Dev: %3.2f%%, Test: %3.2f%%" % (miscl_err_train, miscl_err_dev, miscl_err_test))
    print("")

    




if __name__ == '__main__':
    main()





