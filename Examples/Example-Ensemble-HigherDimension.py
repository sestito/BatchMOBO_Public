# TODO
# Ensemble and sequential are currently setup to maximize the objective function
# This should minimize the objective function
# Leshi's MATLAB code minimizes. See what is different. Maybe the way EI is formed?

# TODO
# if I reverse EI to be tau - mean, would that make it minimize instead of maximize?

# TODO
# Is the sequential targeting MAX or Minimum!?

'''
The current EI method is reversed
    Correct way: https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
    Current way: http://krasserm.github.io/2018/03/21/bayesian-optimization/
Follow what they do in the paper for EI, ect.

For genetic algorithm, use values that Leshi used in the MATLAB code.

'''

import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Ensemble
from pyMOBOS.surrogates import Gaussian
from pyMOBOS.acquisition import Sequential

from pyMOBOS.testfunctions.numerical import DTLZ2

import numpy as np
import matplotlib.pyplot as plt

f = DTLZ2(number_of_parameters=4, number_of_objectives=3)
X_Bounds = f.bounds()
[X, Y] = f.initial_samples(number_of_samples=10, random_style = f.Random.LHS)


#x = np.linspace(-5,6,1000)
#y = f(x)

#number_of_samples = 10
batch_size = 3

#X = np.random.rand(number_of_samples, 1) * 11 - 5
#Y = f(X)
#X_Bounds = np.zeros((2,1))
#X_Bounds[1,:] = 1

#X_Bounds[0,:] = -5
#X_Bounds[1,:] = 6


#plt.plot(x,y,'--')
#plt.plot(X,Y,'.')
#plt.show()

batch_size = 3
for i in range(3):

    surrogate = Gaussian(X, Y) # Could do surrogate.fit() every time
    ac = Sequential(X,Y, X_Bounds, surrogate)
    new_samples = ac()

    #surrogate = Gaussian(X, Y) # Could do surrogate.fit() every time
    #ac = Ensemble(X, Y, X_Bounds, surrogate) #Ensemble is doing EI, not EI combined
    #new_samples = ac(batch_size)
    print(new_samples)
    
    
    #new_samples = ac()
    new_solutions = f(new_samples)

    X = np.append(X, new_samples, axis=0)
    Y = np.append(Y, new_solutions, axis=0)
    
