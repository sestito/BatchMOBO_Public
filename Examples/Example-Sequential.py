import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Sequential
from pyMOBOS.surrogates import Gaussian
from pyMOBOS.testfunctions.numerical import ZDT1

import numpy as np

number_of_samples = 5
number_of_parameters = 4
number_of_objectives = 2

f = ZDT1()
f.set_number_of_parameters(number_of_parameters)
X, Y = f.initial_samples(number_of_samples = number_of_samples, random_style = f.Random.RANDOM)
X_Bounds = f.bounds()



surrogate = Gaussian(X, Y)

ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='EI', mo_method='Euclidean')
print('Expected Improvment Method, Euclidean: ', ac())

ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='EI', mo_method='Mean')
print('Expected Improvment Method, Mean: ', ac())

ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='PI', epsilon=0.1)
print('Probability Improvement Method, Euclidean: ', ac())

ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='UCB', beta=1.5)
print('Upper Confidence Bound Method, Euclidean: ', ac())

ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='LCB', beta=1.5)
print('Lower Confidence Bound Method, Euclidean: ', ac())

# Add in timing
ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='EI', mo_method='Euclidean')
ac.time = True
print('Expected Improvment Method, Euclidean: ', ac())