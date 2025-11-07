import os, sys
from pydoc import ErrorDuringImport


# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import __acquisition_common as acquisition_common

from pyMOBOS.acquisition import Sequential
from pyMOBOS.acquisition import Ensemble
from pyMOBOS.acquisition import Penalty
from pyMOBOS.surrogates import Gaussian
from pyMOBOS.surrogates import GaussianDace

import matplotlib.pyplot as plt

import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel


current_directory = os.path.dirname(os.path.abspath(__file__))
style = os.path.join(current_directory, 'publication.mplstyle')
plt.style.use(style)

def f(x):
    y = x ** 4 - x ** 3 - 2*x**2 - 3*x
    return y

x_real = np.linspace(-2,3,1000)
y_real = f(x_real)

X = np.arange(-2,4).reshape(-1,1)
#X = np.arange(-2,3).reshape(-1,1)
Y = f(X)
X_Bounds = np.array([[-2],[3]])

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
#kernel = ConstantKernel(1.0) * RBF(1.0, length_scale_bounds=(1e-8, 1e8))
kernel = Matern(length_scale_bounds=(1e-20, 1e8))
#kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = None

for i in range(2):
    surrogate = Gaussian(X, Y, kernel=kernel)
    #surrogate = GaussianDace(X, Y)
    
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

    
    
    x_gaus = x_real.reshape((-1,1))
    res_gaus, sigma_gaus = surrogate(x_gaus, True)
    plt.plot(x_gaus, res_gaus)
    plt.legend(['True Solution', 'Original Samples', 'Recommended Point', 'GP Model'])

    scale = 1.96
    ErrorU = res_gaus + scale * sigma_gaus
    ErrorL = res_gaus - scale * sigma_gaus
    plt.fill_between(x_gaus.T[0], ErrorL.T[0], ErrorU.T[0])
    
    
    plt.show()

    res = ac.aquisition(ac, x_gaus)   
    res = acquisition_common.ExpectedImprovementArray(ac, x_gaus) # x_gaus maybe should be np.linspace(min, max, npoints)
    pass
    plt.plot(x_gaus, res)
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



