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


def perceptron_train(d_train, MaxIter):
    """
    Algorithm for training the perceptron (according to CIML Algortithm 5, Chapter 4)
    """
    w = np.zeros(len(d_train[0])-1) # Initialize weight vector
    b = 0 # Initialize bias
    a = 0  # Initialize activiation variable
    # Start the training
    for it in range(50):
        for i in range(len(d_train)): 
            a = np.dot(w,d_train[i][1:]) + b
            # If the prediction is incorrect, update the weight vector
            if ( d_train[i][0] * a <= 0 ):
                w += d_train[i][0] * d_train[i][1:]
                b += d_train[i][0]
    # Norm w
    #w /= np.linalg.norm(w) # gives the Euclidean norm of 
    return(w, b)
        




def main():
    """
    Main function reads in the files from commandline.
    Then starts the routines to train the Perceptron
    """
    # Generating parser to read the filenames (and verbosity level)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--file1", "-f1", type=str, required=False, help="Training data file")
    parser.add_argument("--file2", "-f2", type=str, required=False, help="Test data file")
    parser.add_argument("--verbosity", "-v", type=int, required=False, help="Increase output verbosity (might not be used in this program")
    args = parser.parse_args()
    """
    Uncommend this when using command line please
    
    print( args.file1, args.file2, args.verbosity )
    if ( Path(str(args.file1)).is_file() != True ):
        print(" WARNING: Training file can not be found! ")
        return 0
    if ( Path(str(args.file2)).is_file() != True ):
        print(" WARNING: Test file can not be found! ")
        return 0
    # Read data from ttraining data
    d_train = np.genfromtxt(args.file1)
    d_test = np.genfromtxt(args.file2)
    """
    d_train = np.genfromtxt("A2.1.tsv")
    d_test = np.genfromtxt("A2.2.test.tsv")
    
    # Train the perceptron
    w, b = perceptron_train(d_train, MaxIter = 30)
    #print(w, b)
    is_correct = 0
    isnt_correct = 0
    for i in range(len(d_train)):
        if ( np.sign(np.dot(w,d_train[i][1:]) + b) == d_train[i][0] ):
            is_correct += 1
        else:
            isnt_correct += 1
    print(is_correct)
    print(isnt_correct)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    