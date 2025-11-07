import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Sequential
from pyMOBOS.acquisition import Ensemble
from pyMOBOS.acquisition import Penalty
from pyMOBOS.surrogates import Gaussian

import matplotlib.pyplot as plt

import numpy as np


current_directory = os.path.dirname(os.path.abspath(__file__))
style = os.path.join(current_directory, 'publication.mplstyle')
plt.style.use(style)

def f(x):
    y = x ** 4 - x ** 3 - 2*x**2 - 3*x
    return y

x_real = np.linspace(-2,3,1000)
y_real = f(x_real)

X = np.arange(-2,3).reshape(-1,1)
Y = f(X)
X_Bounds = np.array([[-2],[3]])




for i in range(10):
    surrogate = Gaussian(X, Y)
    
    ac = Sequential(X, Y, X_Bounds, surrogate, aquisition_function='EI', mo_method='Euclidean')
    new_x = ac()

    
    #ac = Penalty(X, Y, X_Bounds, surrogate)
    #new_x = ac(2)
    new_y = f(new_x)

    print('Expected Improvment Method, Euclidean: \n', new_x)

    plt.plot(x_real, y_real, '--')
    plt.plot(X,Y, '*')
    plt.plot(new_x, new_y, '*')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['True Solution', 'Original Samples', 'Recommended Point'])
    plt.show()

    # TODO: Plot the aquisition function!!!!

    #X = np.append(X, [new_x], axis=0)
    #Y = np.append(Y, [new_y], axis=0)

    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    #s = ac.phi(new_x)
    #for ss in s:
    #    ss.plot(-2, 3)
    #
    
    del ac
    del surrogate



