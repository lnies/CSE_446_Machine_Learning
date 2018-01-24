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



def main():
    """
    Main function reads in the files from commandline.
    Then starts the routines to train the Perceptron
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--file1", "-f1", type=str, required=False, help="Training data file")
    args1 = parser.parse_args()
    parser.add_argument("--file2", "-f2", type=str, required=False, help)
    args2 = parser.parse_args()
    print( args1.file1, args2.file2 )


    
    
if __name__ == '__main__':
    main()