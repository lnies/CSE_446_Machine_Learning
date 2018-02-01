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
    plt.figure()
    plt.plot(np.log10(x), training_error, color="blue", label="Training error")
    plt.plot(np.log10(x), test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch (log scale)")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'full_log.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.full.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(0,30)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.early.png', bbox_inches='tight')
    plt.figure()
    plt.plot(x, training_error, color="blue", label="Training error")
    plt.plot(x, test_error, color="red", label="Test error")
    plt.legend()
    plt.xlim(MaxIter-30,MaxIter)
    plt.xlabel("Number of epoch")
    plt.ylabel("Error rate per Epoch in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.late.png', bbox_inches='tight')
    
def plot_inner_loop(inner_train_err, inner_test_err, length, wo):
    x = np.arange(length)
    plt.figure()
    plt.plot(x, inner_train_err, color="blue", label="Training error")
    plt.plot(x, inner_test_err, color="red", label="Test error")
    plt.legend()
    plt.xlabel("Index of observation in epoch")
    plt.ylabel("Error rate per feature in %")
    plt.grid(True)
    plt.savefig("A2.11.train.tsv"+'.inner_loop_at_'+str(wo)+'.png', bbox_inches='tight')

def perceptron_test(d_train, d_test, d_dev, w, b, old_weigths, votes):
    """
    Calculates the error on the trainings set and the error on the testset
    weigthed with the votes of the total of all weight vectors
    """
    error = np.zeros(3)
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
    for i in range(len(d_dev)):
        activation = 0
        for j in range(len(old_weigths)):
            activation += votes[j] * np.sign(np.dot(old_weigths[j],d_dev[i][1:]) + b) 
        if ( np.sign(activation) != d_dev[i][0] ):
            error[2] += 1
    return error


def perceptron_train(d_train, d_test, MaxIter):
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
        # Save last weight vector in the list
        old_weights = np.append(old_weights, w).reshape(len(votes), len(w))
        # Test of the performance of the current perceptron status
        val = perceptron_test(d_train, d_test, d_test, w, b, old_weights, votes)
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

    d_train = np.genfromtxt("A2.2.train.tsv")
    d_test = np.genfromtxt("A2.2.test.tsv")
    MaxIter = 5
    
    # Train the perceptron
    w, b = perceptron_train(d_train, d_test, MaxIter)
    #print(w, b)
    return 0
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    