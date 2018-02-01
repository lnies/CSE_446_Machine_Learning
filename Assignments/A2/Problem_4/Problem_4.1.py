# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:29:05 2018

@author: Lukas
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:32:15 2018

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
# Timer for part 4.1.5
from timeit import default_timer as timer

def plot_10(X):
    """
    Problem 4.1.1: Visualize ten digits
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

def plot_mean(train_set):
    """
    Problem 4.1.3: Visualize the mean of the ten digits
    """
    mu = np.zeros(len(train_set[0][0])).reshape(28,28)
    for i in range(len(train_set[0])):
        mu += train_set[0][i].reshape(28,28)
    mu /= 10
    plt.imshow(mu, cmap='gray')
    print(len(train_set[0]))
    plt.title("Mean of all digits")
    
def calc_cov_vec(train_set):
    """
    Problem 4.1.5: Calculate the covariance matrix based on vector algebra
    """
    # Save data in matrix X and calculate mean and store it in mu
    X = train_set[0]
    n = len(X)
    d = len(X[0])
    sigma = np.zeros(d**2).reshape(d, d)
    Xc = np.zeros(n*d).reshape(d,n)
    Xt = np.transpose(X)
    for i in range(d):
        Xc[i] = np.subtract(Xt[i],(np.mean(Xt[i])))
    Xc = np.transpose(Xc)
    start = timer()
    for i in range(n):
        sigma += np.outer( Xc[i], Xc[i] )
    sigma /= n
    return sigma
    
def calc_cov_mat(train_set):
    """
    Problem 4.1.5: Calculate the covariance matrix based on matrix algebra
    """
    X = train_set[0]
    n = len(X)
    d = len(X[0])
    # First center the matrix X by substracting the mean
    # We do this by transposing and re transposing
    Xc = X - np.mean(X, axis = 0)
    sigma = np.dot(np.transpose(Xc), Xc)
    sigma /= n
    return Xc

def mat_vs_vec(train_set):
    """
    Problem 4.1.5: Compare matrix with vector methods
    """
    start = timer()
    mat = calc_cov_mat(train_set)
    end = timer()
    time_mat = end - start
    print("Process time for matrix method: %3.2fs" % (time_mat))
    start = timer()
    vec = calc_cov_vec(train_set)
    end = timer()
    time_vec = end - start
    print("Process time for vector method: %3.2fs" % (time_vec))
    average_ab_dif = np.sum(np.abs(mat-vec))/((len(train_set[0][0]))**2)
    print("Average absolute difference:", average_ab_dif)
    increase = time_vec * 100 / time_mat 
    print("Increase in runtime for the vector method compared with matrix method: %3.1f" % (increase))

def eigen_analysis(train_set):
    """
    Problem 4.2.1 Calculate the Eigenvalues of the data using SVD
    """
    sigma = calc_cov_mat(train_set)
    u, lamda, v = np.linalg.svd(sigma, full_matrices=1, compute_uv = 1)
    
    print("+++ Information for the calculation of eigenvalues +++")
    print("Eigenvalue Lambda1:", lamda[0])
    print("Eigenvalue Lambda2:", lamda[1])
    print("Eigenvalue Lambda10:", lamda[9])
    print("Eigenvalue Lambda30:", lamda[29])
    print("Eigenvalue Lambda50:", lamda[49])
    print("Sum of eigenvalues:", np.sum(lamda))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    return u, lamda, v

def recon_error(train_set):
    """
    Problem 4.2.2 Plot the fractional reconstruction error for k = 100
    """
    u, lamda, v = eigen_analysis(train_set)
    lamda_sum = np.sum(lamda)
    data = np.array(0)
    data = np.delete(data, 0)
    for k in range(100):
        upstairs = 0
        for i in range(k):
            upstairs += lamda[i]
        error = 1 - ( upstairs ) / ( lamda_sum )
        data = np.append(data, error)
    x = np.arange(1,101)
    plt.figure()
    plt.plot(x, data, label="Fractional error")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Index k")
    plt.ylabel("Fractional reconstruction error")
    plt.show()
    
def variance(train_set):
    """
    Problem 4.2.3 Find the eigenvectors with variance 50% and 80%
    """
    sigma = calc_cov_mat(train_set)
    u, lamda, v = eigen_analysis(train_set)
    lamda_sum = np.sum(lamda)
    summ = 0
    index = 0
    while ( summ < 0.5 * lamda_sum ):
        summ += lamda[index]
        index += 1
    print("The first %i eigenvalues make of 50 percent of the variance" % index)
    summ = 0
    index = 0
    while ( summ < 0.8 * lamda_sum ):
        summ += lamda[index]
        index += 1
    print("The first %i eigenvalues make of 80 percent of the variance" % index)
    
def plot_eigen(train_set):
    """
    Problem 4.2.4: Plot the first 10 eigenvectors
    """
    u, lamda, v = eigen_analysis(train_set)
    print(np.shape(lamda))
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(v[0].reshape(28,28), cmap='gray')
    axarr[0, 0].set_title('Eigenvector 1')
    axarr[0, 1].imshow(v[1].reshape(28,28), cmap='gray')
    axarr[0, 1].set_title('Eigenvector 2')
    axarr[0, 2].imshow(v[2].reshape(28,28), cmap='gray')
    axarr[0, 2].set_title('Eigenvector 3')
    axarr[0, 3].imshow(v[3].reshape(28,28), cmap='gray')
    axarr[0, 3].set_title('Eigenvector 4')
    axarr[0, 4].imshow(v[4].reshape(28,28), cmap='gray')
    axarr[0, 4].set_title('Eigenvector 5')
    axarr[1, 0].imshow(v[5].reshape(28,28), cmap='gray')
    axarr[1, 0].set_title('Eigenvector 6')
    axarr[1, 1].imshow(v[6].reshape(28,28), cmap='gray')
    axarr[1, 1].set_title('Eigenvector 7')
    axarr[1, 2].imshow(v[7].reshape(28,28), cmap='gray')
    axarr[1, 2].set_title('Eigenvector 8')
    axarr[1, 3].imshow(v[8].reshape(28,28), cmap='gray')
    axarr[1, 3].set_title('Eigenvector 9')
    axarr[1, 4].imshow(v[9].reshape(28,28), cmap='gray')
    axarr[1, 4].set_title('Eigenvector 10')

def reconstruct(train_set, k) :
    """
    Problem 4.4.1: Reconstruct images after projecting down to k dimensions
    """
    start = timer()
    X = train_set[0]
    n = len(X)
    d = len(X[0])
    X_rec = np.zeros(n*d).reshape(n,d)
    # First center the matrix X by substracting the mean
    # We do this by transposing and re transposing
    Xc = np.zeros(n*d).reshape(d,n)
    Xt = np.transpose(X)
    for i in range(d):
        Xc[i] = Xt[i] - np.mean(Xt[i])
    Xc = X - np.mean(X, axis = 0)
    # Apply SVD for getting eigenvalues and vectors
    u, lamda, v = eigen_analysis(train_set) 
    
    # Create eigenvector matrix of top k eigenvecors
    U_hat = np.array(0)
    U_hat = np.delete(U_hat, 0)
    for i in range(0, k):
        U_hat = np.append(U_hat, v[i]).reshape(d,i+1)
    # Apply the dimensionality reduction
    X_hat = np.dot(Xc, U_hat) 
    # Now reconstruct data
    # Add the mean back to the data
    X_rec = np.dot(X_hat,np.transpose(U_hat)) + np.mean(X, axis = 0)
    end = timer()
    print("Reconstruction time %f3.2s" % (end-start))
    return X_rec

def plot_reconstructed(train_set, k):
    """
    Problem 4.4.2: Visulaize reconstructed images after projecting down to k dimensions
    """
    X_rec = reconstruct(train_set, k)
    plot_10(X_rec)
    
    
def main():
    """
    Main function to call the previously defined functions
    """
    # Open and save the dataset from NIST
    with gzip.open("mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    # Define dimension
    X = train_set[0]
    n = len(X)
    d = len(X[0])
    """
    Begin of calling the functions
    """
    # Problem part 4.1.1
    #plot_10(X)
    # Problem part 4.1.3
    #plot_mean(train_set)
    # Problem part 4.1.5
    #mat_vs_vec(train_set)
    # Problem 4.2.1
    #eigen_analysis(train_set)
    # Problem 4.2.2
    #recon_error(train_set)
    # Problem 4.2.3
    #variance(train_set)
    # Problem 4.2.4
    #plot_eigen(train_set)
    # Problem 4.4.1
    #reconstruct(train_set, k = 30)
    # Problem 4.4.2
    #plot_reconstructed(train_set, k = 1)
    #plot_reconstructed(train_set, k = 3)
    #plot_reconstructed(train_set, k = 5)
    #plot_reconstructed(train_set, k = 10)
    #plot_reconstructed(train_set, k = 25)
    #plot_reconstructed(train_set, k = 50)
    #plot_reconstructed(train_set, k = 200)
    plot_reconstructed(train_set, k = 500)
    #plot_reconstructed(train_set, k = 784)
    
    #print(np.shape(calc_cov_mat(train_set)))



if __name__ == '__main__':
    main()








