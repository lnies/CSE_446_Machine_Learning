Assignment 2 Lukas Nies

Problem 1: 

/ 

Problem 2:

+ First part without optimization, run from commandline with python:

	python A2_Problem_2.py --test [test.file] --train [train.file] --niter [int: MaxIter]

  Outputs several plots and some information on cout

+ Surprise part (A2_Problem_2_surprise.py) run in editor, does not read args 
  from command line, sets must be in same directory
  Outputs plots with label "10" and some information on command line

Problem 3

+ Run in editor (A2_Problem_3.1.py): same as surprise problem but with slightly different implementation
  - May have a long runtime (depending on MaxIter ) since it's voted perceptron (500 steps take at least 20 mins)

Problem 4

+ Run in editor (A2_Problem_4.1.py), un-comment certain function calls in main() to test for the different
  subproblems. 

Problem 5

/

Problem 6

+ Same as Problem 3 
	- A2_Problem_6.1.py: Run in editor, same A2_Problem_2.1.py (normal perceptron but with PCA, gives set 10 as output 
	- A2_Problem_6.py: Run in editor, same A2_Problem_3.1.py (voted perceptron with PCA, gives set 11 as output) 