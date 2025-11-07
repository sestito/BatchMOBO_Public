import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Penalty, Sequential
from pyMOBOS.surrogates import Gaussian, GaussianDace
from pyMOBOS.testfunctions.numerical import ZDT1, FON, DTLZ1, ZDT3, ZDT2

import matplotlib.pyplot as plt

import numpy as np

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern

current_directory = os.path.dirname(os.path.abspath(__file__))
style = os.path.join(current_directory, 'publication.mplstyle')
plt.style.use(style)

number_of_samples = 20
number_of_parameters = 4
number_of_objectives = 2
batch_size = 1

f = ZDT3()

X = np.random.rand(number_of_samples, number_of_parameters)
Y = f(X)
#Y = DTLZ1(X, number_of_objectives)
#Y = FON(X)
X_Bounds = np.zeros((2,number_of_parameters))
X_Bounds[1,:] = 1

kernel = ConstantKernel(1.0) * RBF(1.0)
kernel = Matern()

for i in range(200):

    surrogate = Gaussian(X, Y, kernel=kernel) # Could do surrogate.fit() every time
    ac = Penalty(X, Y, X_Bounds, surrogate, acquisition_function='EI')
    #ac = Sequential(X, Y, X_Bounds, surrogate, acquisition_function='EI')

    new_samples = ac(5)
    new_solutions = f(new_samples)
    #new_solutions = DTLZ1(new_samples, number_of_objectives)
    #new_solutions = FON(new_samples)

    X = np.append(X, new_samples, axis=0)
    Y = np.append(Y, new_solutions, axis=0)

    # Display the new sample and solution to the user
    print(new_samples)
    print(new_solutions)

    real_solution = f.plot_data()
  
    plt.plot(real_solution[:,0],real_solution[:,1], '.', markersize=1)
    plt.plot(Y[:,0], Y[:,1], '.')
    plt.plot(new_solutions[:,0], new_solutions[:,1], '*')
    #plt.plot([0.051, 0.076, 0.040, 0.036, 0.032], [0.71, 0.56, 1.09, 1.09, 1.24], '*')
    plt.xlabel('Objective 1, $f_{1}$')
    plt.ylabel('Objective 2, $f_{2}$')
    plt.legend(['Real Pareto Front', 'Original Solutions', 'Recommended Solutions'])
    plt.show()
 

print(X)    


