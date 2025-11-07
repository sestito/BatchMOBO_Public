
import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Ensemble
from pyMOBOS.surrogates import Gaussian
from pyMOBOS.acquisition import Sequential

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    #noise = np.random.normal(loc=0, scale=0.1)
    #return (x**2 * np.sin(5*np.pi*x)**6)# + noise
    g = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Return the value of the polynomial
    return g(x) * 0.05

x = np.linspace(-5,6,1000)
y = f(x)

number_of_samples = 10
batch_size = 3

X = np.random.rand(number_of_samples, 1) * 11 - 5
Y = f(X)
X_Bounds = np.zeros((2,1))
X_Bounds[1,:] = 1

X_Bounds[0,:] = -5
X_Bounds[1,:] = 6


plt.plot(x,y,'--')
plt.plot(X,Y,'.')
plt.show()


for i in range(3):

    surrogate = Gaussian(X, Y) # Could do surrogate.fit() every time
    ac = Ensemble(X, Y, X_Bounds, surrogate)
    #ac = Sequential(X,Y, X_Bounds, surrogate)

    new_samples = ac(batch_size)
    #new_samples = ac()
    new_solutions = f(new_samples)

    X = np.append(X, new_samples.reshape(-1,1), axis=0)
    Y = np.append(Y, new_solutions.reshape(-1,1), axis=0)
    
    plt.plot(x,y,'--')
    plt.plot(X,Y,'.')
    plt.plot(new_samples, new_solutions,'.')
    plt.show()
